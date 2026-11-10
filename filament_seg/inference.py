"""Network inference with 8-view (D4) test-time augmentation.

Each network sees the frame at model resolution (1024) in the 8 dihedral
views; the sigmoid maps are mapped back and averaged. Maps are rounded to
float16 per network (the storage precision the submitted run used), the five
fold networks are averaged, and the mean is upsampled bilinearly
(align_corners=False) to the native 2048x2048 grid. Upsampling is linear, so
averaging before or after it is equivalent.

Per frame this yields three native maps, each (2, 2048, 2048) = (foreground, boundary):
  * ``decode``: ConvNeXt ensemble, D4 mean   -> candidates are decoded from it
  * ``single``: ConvNeXt ensemble, identity view only -> agreement features
  * ``vote``:   ResNet34 ensemble, D4 mean   -> vote features
"""
from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from .data import NATIVE, model_input
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
def predict_tta(net: torch.nn.Module, x: torch.Tensor, view_batch: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Sigmoid maps of one network for one frame x (1, 1, S, S): (identity view, D4 mean), float16 (2, S, S)."""
    views = list(d4_views(x))
    outputs = []
    for i in range(0, len(views), view_batch):
        chunk = views[i:i+view_batch]
        probs = torch.sigmoid(net(torch.cat([v for v, _, _ in chunk])).float())
        outputs.extend(undo_view(p[None], k, f) for p, (_, k, f) in zip(probs, chunk))
    total = outputs[0]
    for p in outputs[1:]:
        total = total+p
    single, d4 = outputs[0][0], (total/8)[0]
    return single.cpu().numpy().astype(np.float16), d4.cpu().numpy().astype(np.float16)


def to_native(maps: list[np.ndarray]) -> np.ndarray:
    """Mean of per-network float16 maps (model side) -> native 2048 float32 map (rounded through float16)."""
    total = None
    for m in maps:
        m = m.astype(np.float32)
        total = m if total is None else total+m
    mean = torch.from_numpy(total/len(maps))[None]
    up = F.interpolate(mean, size=(NATIVE, NATIVE), mode='bilinear', align_corners=False)[0].numpy()
    return up.astype(np.float16).astype(np.float32)


class EnsemblePredictor:
    """Loads the segmenter and voter ensembles once and produces the three native maps for a frame."""

    def __init__(self, cfg: dict, weights_dir, segmenters=None, voters=None):
        inf = cfg['inference']
        self.device = pick_device(inf['device'])
        if self.device.type == 'cpu' and inf.get('threads'):
            torch.set_num_threads(int(inf['threads']))
        self.side, self.view_batch = int(inf['side']), int(inf['view_batch'])
        if inf['tta'] != 'd4':
            raise ValueError('Only the 8-view D4 TTA of the submitted pipeline is implemented.')
        names_s = segmenters if segmenters is not None else cfg['weights']['segmenters']
        names_v = voters if voters is not None else cfg['weights']['voters']
        self.segmenters = [load_model(n, weights_dir, self.device) for n in names_s]
        self.voters = [load_model(n, weights_dir, self.device) for n in names_v]

    def __call__(self, image_path) -> dict[str, np.ndarray]:
        x = torch.from_numpy(np.ascontiguousarray(model_input(image_path, self.side)))[None, None].to(self.device)
        single, decode, vote = [], [], []
        for net in self.segmenters:
            s, d = predict_tta(net, x, self.view_batch)
            single.append(s)
            decode.append(d)
        for net in self.voters:
            vote.append(predict_tta(net, x, self.view_batch)[1])
        return {'decode': to_native(decode), 'single': to_native(single), 'vote': to_native(vote)}
