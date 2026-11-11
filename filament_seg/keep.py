"""Keep model of the final pipeline: a bagged GBM on 59 columns per candidate -> keep / reject.

Columns, in this order (``feature_order`` in the pickle):
  x46  the 46 verifier features (``features.FEATURE_NAMES``)
  cnn  the crop-CNN score (mean of the five FZ crop CNNs)
  l2   the six vote features of the context networks (``context.py``)
  dv   the six detector columns (``detector.py``)
Model: 100 HistGradientBoosting regressors (5 bag seeds x 20 bootstrap bags, fitted on the out-of-fold
candidates of all 707 labelled frames); u = clip(mean of the 100 predictions, 0, 1), stored as float32 as in
the submitted run, keep = u >= tau (0.25). The pickle needs scikit-learn 1.9.1, the version it was fitted with.
"""
from __future__ import annotations

import gzip
from pathlib import Path
import pickle

import numpy as np
from threadpoolctl import threadpool_limits

GROUPS = {'x46': 46, 'cnn': 1, 'l2': 6, 'dv': 6}


def load_bundle(path: str | Path) -> dict:
    path = Path(path)
    raw = path.read_bytes()
    return pickle.loads(gzip.decompress(raw) if path.suffix == '.gz' else raw)


class KeepModel:
    def __init__(self, path: str | Path, tau: float = .25):
        bundle = load_bundle(path)
        self.models, self.order, self.tau = bundle['models'], list(bundle['feature_order']), float(tau)
        assert set(self.order) <= set(GROUPS), self.order

    def scores(self, columns: dict[str, np.ndarray]) -> np.ndarray:
        """u (float32) per candidate from {'x46': (n, 46), 'cnn': (n,), 'l2': (n, 6), 'dv': (n, 6)}."""
        parts = []
        for name in self.order:
            v = np.asarray(columns[name])
            v = v if v.ndim == 2 else v[:, None]
            assert v.shape[1] == GROUPS[name], (name, v.shape)
            parts.append(v.astype(np.float64))
        x = np.hstack(parts)
        if not len(x):
            return np.zeros(0, np.float32)
        with threadpool_limits(limits=1, user_api='openmp'):
            return np.clip(np.mean([m.predict(x) for m in self.models], 0), 0, 1).astype(np.float32)

    def keep(self, u: np.ndarray) -> np.ndarray:
        return np.asarray(u, np.float64) >= self.tau
