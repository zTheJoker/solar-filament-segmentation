"""Context networks ("L2"): a second-level U-Net that sees the image together with the first-level maps.

Each of the five context networks is a ConvNeXt-T U-Net with three input channels at 1024x1024:
  * the network input of the frame (``data.model_input``, the same as the segmenters see),
  * the D4 foreground map of the ConvNeXt segmenter of the SAME fold,
  * the D4 foreground map of the ResNet34 voter of the same fold,
both maps quantised to uint8 (round(255 p)) and rescaled as (p - 0.5) / 0.25. Context network k was trained on
out-of-fold maps of fold k, so at inference it is fed fold k's maps, never the ensemble mean.

Every network runs the 8 dihedral views (one view per forward pass, as in the submitted run) under bf16
autocast; its D4 mean is rounded to float16, the five are averaged and upsampled to 2048 (``inference.to_native``).

Out: a native (2, 2048, 2048) map. The verifier uses only its foreground plane, through the same six "vote"
features as the ResNet34 votes (``features.vote_features``), computed on the decoder candidates.
"""
from __future__ import annotations

import numpy as np
import torch

from .features import vote_features
from .inference import predict_tta, to_native


def context_input(x: torch.Tensor, cnx_fold: np.ndarray, r34_fold: np.ndarray) -> torch.Tensor:
    """(1, 3, S, S) input of one context network: frame input x (1, 1, S, S) + fold maps (2, S, S) float16 each."""
    maps = np.stack([np.round(cnx_fold[0].astype(np.float32)*255), np.round(r34_fold[0].astype(np.float32)*255)])
    maps = (maps.astype(np.uint8).astype(np.float32)/255-.5)/.25
    return torch.cat([x, torch.from_numpy(maps)[None].to(x.device)], 1)


class ContextVoter:
    """The five context networks; ``maps`` gives the native context map, ``votes`` its six vote features."""

    def __init__(self, nets, view_batch: int = 1, autocast: str | None = 'bf16'):
        self.nets = list(nets)
        self.view_batch = int(view_batch)
        self.autocast = {'bf16': torch.bfloat16, None: None}[autocast]

    def maps(self, x: torch.Tensor, cnx_folds: list[np.ndarray], r34_folds: list[np.ndarray]) -> np.ndarray:
        """Native (2, 2048, 2048) mean context map. x: (1, 1, S, S) frame input; per-fold (2, S, S) float16 maps."""
        assert len(cnx_folds) == len(r34_folds) == len(self.nets), 'one ConvNeXt and one ResNet34 map per context network'
        out = []
        for net, c, r in zip(self.nets, cnx_folds, r34_folds):
            _, d4 = predict_tta(net, context_input(x, c, r), self.view_batch, autocast=self.autocast)
            out.append(d4.numpy().astype(np.float16))
        return to_native(out)

    @staticmethod
    def votes(labels: np.ndarray, context_map: np.ndarray) -> np.ndarray:
        """(n_candidates, 6) vote features of the context map on the decoder candidates."""
        return vote_features(labels, context_map)
