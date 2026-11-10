"""Network inference with 8-view (D4) test-time augmentation and map fusion.

Base networks (ConvNeXt segmenters, ResNet34 voters) see the frame at model
resolution (1024) in the 8 dihedral views; the sigmoid maps are mapped back
and averaged. Maps are rounded to float16 per network (the storage precision
the submitted runs used), the five fold networks are averaged, and the mean is
upsampled bilinearly (align_corners=False) to the native 2048x2048 grid.

FZ mode adds five small-filament specialists: the same ConvNeXt U-Net,
fine-tuned on native-resolution crops, run on the full 2048 frame (no resize)
under bf16 autocast, foreground channel only, quantised to uint8 per network
(again the stored precision of the submitted run). Their mean q is fused with
the base foreground p as (1 - w) p + w q with w = 0.5; the boundary channel
stays the base one.

Per frame this yields three native maps, each (2, 2048, 2048) = (foreground, boundary):
  * ``decode``: D4 map (fused in FZ mode)            -> candidates are decoded from it
  * ``single``: identity-view map (fused in FZ mode)  -> agreement features, crop CNN
  * ``vote``:   ResNet34 ensemble, D4 mean            -> vote features
"""
from __future__ import annotations

import contextlib

import numpy as np
import torch
from torch.nn import functional as F

from .data import NATIVE, model_input, normalize_image, read_gray
from .models import load_model


def pick_device(name: str = 'auto') -> torch.device:
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)


def d4_views(x: torch.Tensor):
    """The 8 dihedral views of a (B, C, H, W) batch as (view, k_rot90, flipped)."""
    for k in range(4):
        r = torch.rot90(x, k, (-2, -1))
        yield r, k, False
        yield torch.flip(r, (-1,)), k, True


def undo_view(p: torch.Tensor, k: int, flipped: bool) -> torch.Tensor:
    if flipped:
        p = torch.flip(p, (-1,))
    return torch.rot90(p, -k, (-2, -1))


@torch.inference_mode()
def predict_tta(net: torch.nn.Module, x: torch.Tensor, view_batch: int = 8, channels=slice(0, 2),
                autocast: torch.dtype | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Sigmoid maps of one network for one frame x (1, 1, S, S): (identity view, D4 mean), each (C, S, S) float32."""
    views = list(d4_views(x))
    outputs = []
    for i in range(0, len(views), view_batch):
        chunk = views[i:i+view_batch]
        ctx = torch.autocast(x.device.type, dtype=autocast) if autocast else contextlib.nullcontext()
        with ctx:
            logits = net(torch.cat([v for v, _, _ in chunk]))[:, channels]
        probs = torch.sigmoid(logits.float())
        outputs.extend(undo_view(p[None], k, f) for p, (_, k, f) in zip(probs, chunk))
    total = outputs[0]
    for p in outputs[1:]:
        total = total+p
    return outputs[0][0].cpu(), (total/8)[0].cpu()


def to_native(maps: list[np.ndarray]) -> np.ndarray:
    """Mean of per-network float16 maps (model side) -> native 2048 float32 map (rounded through float16)."""
    total = None
    for m in maps:
        m = m.astype(np.float32)
        total = m if total is None else total+m
    mean = torch.from_numpy(total/len(maps))[None]
    up = F.interpolate(mean, size=(NATIVE, NATIVE), mode='bilinear', align_corners=False)[0].numpy()
    return up.astype(np.float16).astype(np.float32)


def specialist_mean(maps_uint8: list[np.ndarray]) -> np.ndarray:
    """Mean of per-specialist uint8 foreground maps -> float32 in [0, 1]."""
    total = None
    for m in maps_uint8:
        m = m.astype(np.float32)
        total = m if total is None else total+m
    return total/len(maps_uint8)/255.


def fuse(base: np.ndarray, q: np.ndarray, weight: float) -> np.ndarray:
    """Fused (2, H, W) map: foreground (1-w)*base + w*q, base boundary; rounded through float16 like the stored maps."""
    fg = (1-weight)*base[0]+weight*q if weight != .5 else (base[0]+q)/2
    return np.stack([fg, base[1]]).astype(np.float16).astype(np.float32)


class EnsemblePredictor:
    """Holds the network ensembles and produces the three native maps for a frame."""

    def __init__(self, cfg: dict, segmenters, voters, specialists=(), device=None):
        inf = cfg['inference']
        self.device = device or pick_device(inf['device'])
        if self.device.type == 'cpu' and inf.get('threads'):
            torch.set_num_threads(int(inf['threads']))
        if inf['tta'] != 'd4':
            raise ValueError('Only the 8-view D4 TTA of the submitted pipelines is implemented.')
        self.side, self.view_batch = int(inf['side']), int(inf['view_batch'])
        self.segmenters, self.voters, self.specialists = list(segmenters), list(voters), list(specialists)
        spec = inf.get('specialist', {})
        self.spec_batch = int(spec.get('view_batch', 1))
        self.spec_autocast = {'bf16': torch.bfloat16, None: None}[spec.get('autocast')]
        self.fusion_weight = float(spec.get('fusion_weight', .5))

    @classmethod
    def from_weights(cls, cfg: dict, weights_dir, segmenters=None, voters=None, specialists=None):
        """Load the networks named in the config (or the given name lists) from weights/."""
        device = pick_device(cfg['inference']['device'])
        w = cfg['weights']
        load = lambda names: [load_model(n, weights_dir, device) for n in names]
        return cls(cfg, load(segmenters if segmenters is not None else w['segmenters']),
                   load(voters if voters is not None else w['voters']),
                   load(specialists if specialists is not None else w.get('specialists', [])), device)

    def specialist_maps(self, gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Mean specialist foreground at native resolution: (identity view, D4 mean)."""
        x = torch.from_numpy(normalize_image(gray))[None, None].to(self.device)
        single, d4 = [], []
        for net in self.specialists:
            s, d = predict_tta(net, x, self.spec_batch, slice(0, 1), self.spec_autocast)
            single.append(np.round(s[0].numpy()*255).astype(np.uint8))
            d4.append(np.round(d[0].numpy()*255).astype(np.uint8))
        return specialist_mean(single), specialist_mean(d4)

    def __call__(self, image_path) -> dict[str, np.ndarray]:
        x = torch.from_numpy(np.ascontiguousarray(model_input(image_path, self.side)))[None, None].to(self.device)
        single, decode, vote = [], [], []
        for net in self.segmenters:
            s, d = predict_tta(net, x, self.view_batch)
            single.append(s.numpy().astype(np.float16))
            decode.append(d.numpy().astype(np.float16))
        for net in self.voters:
            vote.append(predict_tta(net, x, self.view_batch)[1].numpy().astype(np.float16))
        maps = {'decode': to_native(decode), 'single': to_native(single), 'vote': to_native(vote)}
        if self.specialists:
            q_single, q_decode = self.specialist_maps(read_gray(image_path))
            maps['decode'] = fuse(maps['decode'], q_decode, self.fusion_weight)
            maps['single'] = fuse(maps['single'], q_single, self.fusion_weight)
        return maps
