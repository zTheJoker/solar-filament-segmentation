"""Candidate verifier: crop CNN score + 46 features -> bagged GBM -> keep / reject.

The verifier regresses u = mean over annotators of the matched IoU of a
candidate (0 when unmatched), i.e. the candidate's share of the PQ numerator.
Keeping a candidate raises PQ only if u exceeds about PQ/2, which is why the
frozen threshold tau = 0.25 sits close to half the pipeline's PQ.

Crop CNN input: a square crop of side max(96, longest bbox side + 32) centred
on the candidate, resized to 128x128 (INTER_AREA), 5 uint8 channels:
local z-scored image, decode-map fg, single-view fg, this candidate's mask,
all other candidates. The score is the mean over the 5 fold CNNs of a
4-rotation TTA.
"""
from __future__ import annotations

import gzip
from pathlib import Path
import pickle

import cv2
import numpy as np
import torch
from scipy import ndimage as ndi
from threadpoolctl import threadpool_limits

from .models import load_model

CROP = 128


def candidate_crops(gray: np.ndarray, labels: np.ndarray, decode: np.ndarray, single: np.ndarray,
                    crop: int = CROP) -> tuple[np.ndarray, np.ndarray]:
    """(n, 5, crop, crop) uint8 crops and (n,) float32 log crop sides, one per candidate."""
    gray = np.asarray(gray, np.float32)
    lab = labels.astype(np.int32)
    size = lab.shape[0]
    crops, sides = [], []
    for i, box in enumerate(ndi.find_objects(lab)):
        y0, y1, x0, x1 = box[0].start, box[0].stop, box[1].start, box[1].stop
        side = max(96, max(y1-y0, x1-x0)+32)
        a0, b0 = (y0+y1)//2-side//2, (x0+x1)//2-side//2

        def cut(arr):
            out = np.zeros((side, side), np.float32)
            ys, xs = max(0, a0), max(0, b0)
            ye, xe = min(size, a0+side), min(size, b0+side)
            out[ys-a0:ye-a0, xs-b0:xe-b0] = arr[ys:ye, xs:xe]
            return cv2.resize(out, (crop, crop), interpolation=cv2.INTER_AREA)
        g = cut(gray)
        z = np.clip((g-g.mean())/(g.std()+1)*40+128, 0, 255)
        channels = [z, cut(decode[0])*255, cut(single[0])*255, cut((lab == i+1).astype(np.float32))*255,
                    cut(((lab > 0) & (lab != i+1)).astype(np.float32))*255]
        crops.append(np.stack(channels).round().astype(np.uint8))
        sides.append(np.log(side))
    return np.asarray(crops, np.uint8).reshape(-1, 5, crop, crop), np.asarray(sides, np.float32)


class Verifier:
    """Holds the fold crop CNNs and the bagged GBM; scores candidates and decides which to keep."""

    def __init__(self, cfg: dict, weights_dir, device: torch.device, crop_cnns=None, gbm=None):
        v = cfg['verifier']
        self.tau, self.batch, self.device = float(v['tau']), int(v['cnn_batch']), device
        names = crop_cnns if crop_cnns is not None else cfg['weights']['crop_cnns']
        self.cnns = [load_model(n, weights_dir, device) for n in names]
        gbm_path = Path(weights_dir)/(gbm or cfg['weights']['gbm'])
        raw = gbm_path.read_bytes()
        bundle = pickle.loads(gzip.decompress(raw) if gbm_path.suffix == '.gz' else raw)
        self.gbm_models, self.feature_cols = bundle['models'], list(bundle['cols'])

    @torch.inference_mode()
    def cnn_scores(self, crops: np.ndarray, sides: np.ndarray) -> np.ndarray:
        """Mean over fold CNNs of the 4-rotation TTA probability for u (bf16 autocast, as in training)."""
        if not len(crops):
            return np.zeros(0)
        xs, ss = torch.from_numpy(crops), torch.from_numpy(sides)
        per_model = []
        for net in self.cnns:
            out = []
            for i in range(0, len(xs), self.batch):
                x = xs[i:i+self.batch].to(self.device).float()/255
                s = ss[i:i+self.batch].to(self.device)
                acc = 0
                with torch.autocast(self.device.type, dtype=torch.bfloat16):
                    for k in range(4):
                        acc = acc+torch.sigmoid(net(torch.rot90(x, k, (-2, -1)), s).float())
                out.append((acc/4).cpu())
            per_model.append(torch.cat(out).numpy()[:, 0])
        return np.mean(per_model, 0)

    def gbm_scores(self, features: np.ndarray, cnn: np.ndarray) -> np.ndarray:
        """u predicted by the mean of the bagged GBMs, clipped to [0, 1]."""
        if not len(features):
            return np.zeros(0)
        x = np.hstack([features[:, self.feature_cols], cnn[:, None]])
        # A frame has only a few dozen candidates: one OpenMP thread is fastest and avoids
        # thread over-subscription next to PyTorch's own pool (the result does not depend on it).
        with threadpool_limits(limits=1, user_api='openmp'):
            return np.clip(np.mean([m.predict(x) for m in self.gbm_models], 0), 0, 1)

    def keep(self, u: np.ndarray) -> np.ndarray:
        return u >= self.tau
