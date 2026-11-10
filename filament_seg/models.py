"""Network definitions and weight loading.

* ``ConvNeXtUNet`` - the segmenter: timm ConvNeXt-T encoder (1-channel stem)
  with a GroupNorm U-Net decoder and two shallow full/half-resolution branches,
  because the stride-4 ConvNeXt stem loses thin filament detail.
* ``ResNet34UNet`` - a second, independently trained U-Net family whose maps
  only feed the verifier's "vote" features.
* ``CropVerifierCNN`` - ResNet18 on 5-channel 128x128 crops around a
  candidate, plus log crop size; predicts the candidate's expected IoU share.

Both U-Nets output two logit planes: filament foreground and instance boundary.

Weights live in ``weights/`` as safetensors shards (GitHub refuses files over
100 MB). ``weights/manifest.json`` lists the shards of every model and a
SHA-256 of the reassembled fp32 state dict, which ``load_model`` verifies.
The U-Nets are stored in fp32. The crop CNNs only ever run under bf16 autocast,
which rounds their convolution/linear weights to bf16 on every call, so those
tensors are stored in bf16 (exactly what autocast computes with) and widened
back to fp32 on load; BatchNorm tensors stay fp32.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import timm
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet34


def _up_cat(z, skip):
    return torch.cat([F.interpolate(z, size=skip.shape[-2:], mode='bilinear', align_corners=False), skip], 1)


class GNBlock(nn.Sequential):
    """Two 3x3 convolutions, each followed by GroupNorm and GELU."""

    def __init__(self, cin, cout, groups=8):
        super().__init__(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.GroupNorm(groups, cout), nn.GELU(),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.GroupNorm(groups, cout), nn.GELU())


class ConvNeXtUNet(nn.Module):
    """ConvNeXt-T U-Net with stride-1 and stride-2 shallow skips; output (B, 2, H, W) logits."""

    def __init__(self, encoder='convnext_tiny.fb_in22k_ft_in1k', pretrained=False):
        super().__init__()
        self.encoder = timm.create_model(encoder, pretrained=pretrained, features_only=True, in_chans=1)
        c = self.encoder.feature_info.channels()  # 96, 192, 384, 768 at strides 4, 8, 16, 32
        self.s1 = GNBlock(1, 16)
        self.s2 = nn.Sequential(nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False), nn.GroupNorm(8, 32), nn.GELU(),
                                GNBlock(32, 32))
        self.d3 = GNBlock(c[3]+c[2], 256)
        self.d2 = GNBlock(256+c[1], 128)
        self.d1 = GNBlock(128+c[0], 64)
        self.d0 = GNBlock(64+32, 32)
        self.full = GNBlock(32+16, 24)
        self.head = nn.Conv2d(24, 2, 1)

    def forward(self, x):
        f1 = self.s1(x)
        f2 = self.s2(f1)
        a, b, c, d = self.encoder(x)
        z = self.d3(_up_cat(d, c))
        z = self.d2(_up_cat(z, b))
        z = self.d1(_up_cat(z, a))
        z = self.d0(_up_cat(z, f2))
        return self.head(self.full(_up_cat(z, f1)))


class BNBlock(nn.Sequential):
    """Two 3x3 convolutions, each followed by BatchNorm and ReLU."""

    def __init__(self, cin, cout):
        super().__init__(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class ResNet34UNet(nn.Module):
    """Grayscale ResNet34 U-Net; output (B, 2, H, W) logits."""

    def __init__(self, pretrained=False):
        super().__init__()
        from torchvision.models import ResNet34_Weights
        encoder = resnet34(weights=ResNet34_Weights.DEFAULT if pretrained else None)
        rgb_stem = encoder.conv1
        encoder.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
        if pretrained:  # grayscale stem = sum of the ImageNet RGB filters
            with torch.no_grad():
                encoder.conv1.weight.copy_(rgb_stem.weight.sum(dim=1, keepdim=True))
        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.pool = encoder.maxpool
        self.e1, self.e2, self.e3, self.e4 = encoder.layer1, encoder.layer2, encoder.layer3, encoder.layer4
        self.d3 = BNBlock(512+256, 256)
        self.d2 = BNBlock(256+128, 128)
        self.d1 = BNBlock(128+64, 64)
        self.d0 = BNBlock(64+64, 32)
        self.full = BNBlock(32, 24)
        self.head = nn.Conv2d(24, 2, 1)

    def forward(self, x):
        size = x.shape[-2:]
        a = self.stem(x)
        b = self.e1(self.pool(a))
        c = self.e2(b)
        d = self.e3(c)
        e = self.e4(d)
        z = self.d3(_up_cat(e, d))
        z = self.d2(_up_cat(z, c))
        z = self.d1(_up_cat(z, b))
        z = self.d0(_up_cat(z, a))
        return self.head(self.full(F.interpolate(z, size=size, mode='bilinear', align_corners=False)))


class CropVerifierCNN(nn.Module):
    """ResNet18 (5 input channels) + log crop side -> 2 logits: [expected IoU share u, match fraction]."""

    def __init__(self, pretrained=False):
        super().__init__()
        self.body = timm.create_model('resnet18.a1_in1k', pretrained=pretrained, in_chans=5, num_classes=0)
        self.head = nn.Sequential(nn.Linear(513, 128), nn.ReLU(), nn.Linear(128, 2))

    def forward(self, x, log_side):
        return self.head(torch.cat([self.body(x), log_side[:, None]], 1))


ARCHITECTURES = {'ConvNeXtUNet': ConvNeXtUNet, 'ResNet34UNet': ResNet34UNet, 'CropVerifierCNN': CropVerifierCNN}


def state_dict_sha256(state: dict) -> str:
    """Content hash of a state dict: key, dtype, shape and raw bytes of every tensor, in sorted key order."""
    h = hashlib.sha256()
    for key in sorted(state):
        t = state[key].detach().cpu().contiguous()
        h.update(f'{key}|{t.dtype}|{tuple(t.shape)}|'.encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def read_manifest(weights_dir: str | Path) -> dict:
    return json.loads((Path(weights_dir)/'manifest.json').read_text())


def load_state(name: str, weights_dir: str | Path) -> tuple[dict, dict]:
    """Reassemble the shards of ``name`` into one fp32 state dict (bf16-stored tensors widened) and verify its hash."""
    from safetensors.torch import load_file
    entry = read_manifest(weights_dir)[name]
    state = {}
    for shard in entry['files']:
        part = load_file(str(Path(weights_dir)/shard))
        assert not set(part) & set(state), f'{shard}: duplicate keys across shards'
        state.update(part)
    state = {k: v.float() if v.dtype == torch.bfloat16 else v for k, v in state.items()}
    if state_dict_sha256(state) != entry['state_sha256']:
        raise ValueError(f'{name}: reassembled weights do not match the manifest hash')
    return state, entry


def load_model(name: str, weights_dir: str | Path, device='cpu') -> nn.Module:
    """Build the architecture named in the manifest, load its verified weights, return it in eval mode."""
    state, entry = load_state(name, weights_dir)
    kwargs = {'encoder': entry['encoder']} if entry['architecture'] == 'ConvNeXtUNet' else {}
    net = ARCHITECTURES[entry['architecture']](pretrained=False, **kwargs)
    net.load_state_dict(state, strict=True)
    return net.to(device).eval()
