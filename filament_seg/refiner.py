"""Instance refiner: re-segments each candidate in a rotated crop at native resolution.

Which candidates: every candidate whose FZ verifier score u_FZ >= ``u_min`` (0.10), whether kept or not; the
paste-back step decides which refined masks are used.

Crop of a candidate (seed = its nz1 outline):
  * minimum-area rectangle of the seed; the crop is rotated so that the long axis is horizontal;
  * padding max(0.5 L, 32 px) on each side along the axis, max(0.25 W, 24 px) across;
  * native scale whenever the padded rectangle fits bucket A (128 x 384) or B (192 x 768), otherwise scaled down
    into B. The crop is first sampled with a margin of (16, 32) px (the training-time augmentation margin) and the
    network sees its centre, exactly as in training.
Input channels (uint8, then ``prep_input``):
  0 grayscale (``data.normalize_image`` of the native frame, clipped to +-2)
  1 fused D4 foreground map of dense fold k for refiner k (per-fold maps, see ``inference``)
  2 the seed
  3 the other candidates with u_FZ >= u_min, each weighted by its u_FZ
Network: five ``RefineUNet`` (one per fold), four flips each (identity, horizontal, vertical, both) under bf16
autocast; the sigmoid outputs are averaged over flips, summed over the five folds and divided by five.

Out per candidate: the refined probability on the native bounding box of the crop footprint, uint8 (x 255)
after resampling the crop-frame probability (itself quantised to uint8) back onto the native grid:
{candidate index: (y0, y1, x0, x1, uint8 [y1-y0, x1-x0])}.
"""
from __future__ import annotations

import contextlib

import cv2
import numpy as np
import torch
from scipy import ndimage as ndi

BUCKETS = {'A': {'inner': (128, 384), 'margin': (16, 32)}, 'B': {'inner': (192, 768), 'margin': (16, 32)}}
PAD_ALONG = (.5, 32.)
PAD_ACROSS = (.25, 24.)
GRAY_CLIP = 2.
NATIVE = 2048


def q8(a: np.ndarray) -> np.ndarray:
    return np.clip(np.round(a*255), 0, 255).astype(np.uint8)


def gray_q8(a: np.ndarray) -> np.ndarray:
    return q8((np.clip(a, -GRAY_CLIP, GRAY_CLIP)+GRAY_CLIP)/(2*GRAY_CLIP))


def seed_rect(mask: np.ndarray, y0: int = 0, x0: int = 0):
    """Minimum-area rectangle of a seed: centre (x, y), unit long-axis direction (dx >= 0), length L, width W."""
    ys, xs = np.nonzero(mask)
    pts = np.stack([xs+x0, ys+y0], 1).astype(np.float32)
    if len(pts) < 3:
        return (float(pts[:, 0].mean()), float(pts[:, 1].mean())), (1., 0.), 1., 1.
    box = cv2.boxPoints(cv2.minAreaRect(cv2.convexHull(pts)))
    e0, e1 = box[1]-box[0], box[2]-box[1]
    l0, l1 = float(np.hypot(*e0)), float(np.hypot(*e1))
    if l0 >= l1:
        d, length, width = e0/max(l0, 1e-6), l0, l1
    else:
        d, length, width = e1/max(l1, 1e-6), l1, l0
    if length < 1e-3:
        d = np.array([1., 0.], np.float32)
    if d[0] < 0 or (d[0] == 0 and d[1] < 0):
        d = -d
    c = box.mean(0)
    return (float(c[0]), float(c[1])), (float(d[0]), float(d[1])), length+1., width+1.


def choose_bucket(length: float, width: float) -> tuple[str, float]:
    """Native scale whenever the padded rectangle fits a bucket; scale down into B only when it fits neither."""
    need_along = length+2*max(PAD_ALONG[0]*length, PAD_ALONG[1])
    need_across = width+2*max(PAD_ACROSS[0]*width, PAD_ACROSS[1])
    for name in ('A', 'B'):
        h, w = BUCKETS[name]['inner']
        if need_along <= w and need_across <= h:
            return name, 1.
    h, w = BUCKETS['B']['inner']
    return 'B', float(min(w/need_along, h/need_across))


def affine(center, d, scale: float, out_hw) -> np.ndarray:
    """2x3 matrix mapping native (x, y) pixel centres to crop (u, v) pixel centres; u runs along the long axis."""
    (cx, cy), (dx, dy) = center, d
    h, w = out_hw
    return np.array([[scale*dx, scale*dy, (w-1)/2-scale*(cx*dx+cy*dy)],
                     [-scale*dy, scale*dx, (h-1)/2-scale*(-cx*dy+cy*dx)]], np.float64)


def crop_roi(M: np.ndarray, out_hw, pad: int = 3):
    """Native bounding box (y0, y1, x0, x1; clipped) of a crop's footprint, and M re-based to that box."""
    h, w = out_hw
    inv = np.linalg.inv(np.vstack([M, [0, 0, 1]]))
    cor = (inv @ np.array([[-.5, -.5, 1], [w-.5, -.5, 1], [-.5, h-.5, 1], [w-.5, h-.5, 1]]).T)[:2]
    x0 = int(max(0, np.floor(cor[0].min())-pad))
    x1 = int(min(NATIVE, np.ceil(cor[0].max())+pad+1))
    y0 = int(max(0, np.floor(cor[1].min())-pad))
    y1 = int(min(NATIVE, np.ceil(cor[1].max())+pad+1))
    Mr = M.copy()
    Mr[:, 2] += M[:, 0]*x0+M[:, 1]*y0
    return (y0, y1, x0, x1), Mr


def warp(src_roi: np.ndarray, Mr: np.ndarray, out_hw, border: float = 0.) -> np.ndarray:
    h, w = out_hw
    return cv2.warpAffine(np.ascontiguousarray(src_roi, np.float32), Mr, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=float(border))


def unwarp(crop: np.ndarray, Mr: np.ndarray, roi_hw) -> np.ndarray:
    """Sample a crop-frame map back on the native box grid (bilinear, 0 outside the crop)."""
    h, w = roi_hw
    return cv2.warpAffine(np.ascontiguousarray(crop, np.float32), Mr, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0.)


def prep_input(x8: torch.Tensor) -> torch.Tensor:
    """uint8 crop channels (B, 4, H, W) -> network input: grayscale back to its normalised scale, others in [0, 1]."""
    x = x8.float()/255
    return torch.cat([x[:, :1]*4-2, x[:, 1:]], 1)


def build_crops(gray_norm: np.ndarray, seeds: np.ndarray, u_fz: np.ndarray, fused_folds: list[np.ndarray],
                u_min: float = .10) -> list[dict]:
    """One crop record per candidate with u_FZ >= u_min, in candidate order.

    gray_norm: ``data.normalize_image`` of the native frame; seeds: nz1 label image of all candidates (label j+1);
    fused_folds: per-fold native fused foreground maps. Each record holds the bucket, the affine matrix of the
    margin crop and the uint8 channels: ``x`` (3, H, W) = gray, seed, neighbours and ``pf`` (folds, H, W).
    """
    u = np.asarray(u_fz, np.float64)
    n = len(u)
    weights = np.r_[0., np.where(u >= u_min, u, 0.)].astype(np.float32)
    objects = ndi.find_objects(seeds, max_label=n)
    records = []
    for j in range(n):
        if u[j] < u_min:
            continue
        b = objects[j]
        center, d, length, width = seed_rect(seeds[b] == j+1, b[0].start, b[1].start)
        bucket, scale = choose_bucket(length, width)
        (ih, iw), (my, mx) = BUCKETS[bucket]['inner'], BUCKETS[bucket]['margin']
        shw = (ih+2*my, iw+2*mx)
        M = affine(center, d, scale, shw)
        (y0, y1, x0, x1), Mr = crop_roi(M, shw)
        local = seeds[y0:y1, x0:x1]
        seed = local == j+1
        neighbours = weights[local]
        neighbours[seed] = 0
        x = np.stack([gray_q8(warp(gray_norm[y0:y1, x0:x1], Mr, shw)), q8(warp(seed, Mr, shw)), q8(warp(neighbours, Mr, shw))])
        pf = np.stack([q8(warp(m[y0:y1, x0:x1], Mr, shw)) for m in fused_folds])
        records.append({'cand': j, 'bucket': bucket, 'M': M, 'x': x, 'pf': pf})
    return records


class Refiner:
    """The fold refiners; ``refine`` returns {candidate: (y0, y1, x0, x1, uint8 probability)} for one frame."""

    def __init__(self, nets, device: torch.device, batch=None, autocast: str | None = 'bf16', flips: bool = True,
                 u_min: float = .10):
        self.nets, self.device = list(nets), device
        self.batch = {'A': 48, 'B': 12, **(batch or {})}
        self.autocast = {'bf16': torch.bfloat16, None: None}[autocast]
        self.flips = [(), (-1,), (-2,), (-2, -1)] if flips else [()]
        self.u_min = float(u_min)
        if device.type == 'cuda':
            self.nets = [net.to(memory_format=torch.channels_last) for net in self.nets]

    @torch.inference_mode()
    def _forward(self, net, x8: np.ndarray) -> np.ndarray:
        x = prep_input(torch.from_numpy(x8).to(self.device))
        ctx = torch.autocast(self.device.type, dtype=self.autocast) if self.autocast else contextlib.nullcontext()
        acc = 0
        with ctx:
            for dims in self.flips:
                z = net(torch.flip(x, dims) if dims else x).float().sigmoid()
                acc = acc+(torch.flip(z, dims) if dims else z)
        return (acc/len(self.flips))[:, 0].cpu().numpy()

    def probabilities(self, records: list[dict]) -> list[np.ndarray]:
        """Crop-frame refined probability (inner size, float32) per record: fold outputs summed, divided by folds."""
        total = [0]*len(records)
        for k, net in enumerate(self.nets):
            for bucket in 'AB':
                idx = [i for i, r in enumerate(records) if r['bucket'] == bucket]
                (h, w), (my, mx) = BUCKETS[bucket]['inner'], BUCKETS[bucket]['margin']
                for s in range(0, len(idx), self.batch[bucket]):
                    chunk = idx[s:s+self.batch[bucket]]
                    x8 = np.stack([np.concatenate([records[i]['x'][:1], records[i]['pf'][k:k+1], records[i]['x'][1:]])
                                   for i in chunk])[:, :, my:my+h, mx:mx+w]
                    for i, p in zip(chunk, self._forward(net, np.ascontiguousarray(x8))):
                        total[i] = total[i]+p
        return [t/len(self.nets) for t in total]

    def refine(self, gray_norm, seeds, u_fz, fused_folds) -> dict[int, tuple]:
        assert len(fused_folds) == len(self.nets), 'refiner k needs the fused map of dense fold k'
        records = build_crops(gray_norm, seeds, u_fz, fused_folds, self.u_min)
        out = {}
        for r, p in zip(records, self.probabilities(records)):
            (h, w), (my, mx) = BUCKETS[r['bucket']]['inner'], BUCKETS[r['bucket']]['margin']
            Mi = r['M'].copy()
            Mi[0, 2] -= mx
            Mi[1, 2] -= my
            (y0, y1, x0, x1), Mr = crop_roi(Mi, (h, w), pad=1)
            out[r['cand']] = (y0, y1, x0, x1, q8(unwarp(q8(p).astype(np.float32)/255, Mr, (y1-y0, x1-x0))))
        return out
