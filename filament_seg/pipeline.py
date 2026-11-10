"""End-to-end frame pipeline: networks -> candidates -> verifier -> post-processing -> RLE.

    image --(EnsemblePredictor)--> decode / single / vote maps (2048^2)
          --(candidate_labels)----> candidate instances (fg >= 0.4, area >= 150)
          --(features + crop CNN + GBM)--> u per candidate, keep = u >= tau
          --(postprocess steps)---> expected-IoU outlines, rejected removed
          --(encode_label_image)--> one RLE row per kept instance

Only the test image itself is used at inference (no file-name date, no
metadata, no other frame).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import time

import numpy as np

from .candidates import CandidateSettings, candidate_labels
from .config import resolve
from .data import read_gray
from .features import candidate_features
from .inference import EnsemblePredictor
from .postprocess import InstanceState, run_steps
from .submission import encode_label_image
from .verifier import Verifier, candidate_crops


@dataclass
class FrameResult:
    key: str
    rles: list[str]
    instances: np.ndarray          # final label image (1..K)
    candidates: np.ndarray         # candidate label image (1..n)
    features: np.ndarray           # (n, 46)
    cnn_u: np.ndarray              # (n,) crop-CNN score
    u: np.ndarray                  # (n,) verifier score
    keep: np.ndarray               # (n,) bool
    seconds: dict = field(default_factory=dict)


class Pipeline:
    """The submitted pipeline. Model lists can be overridden (the notebook's held-out evaluation uses one fold)."""

    def __init__(self, cfg: dict, segmenters=None, voters=None, crop_cnns=None, gbm=None):
        weights_dir = resolve(cfg['weights']['dir'])
        self.predictor = EnsemblePredictor(cfg, weights_dir, segmenters, voters)
        self.verifier = Verifier(cfg, weights_dir, self.predictor.device, crop_cnns, gbm)
        self.settings = CandidateSettings.from_config(cfg)
        self.steps = cfg['postprocess']['steps']

    def process(self, key: str, gray: np.ndarray, maps: dict[str, np.ndarray]) -> FrameResult:
        """Everything after the networks, for one frame."""
        t0 = time.perf_counter()
        labels = candidate_labels(maps['decode'][0], self.settings)
        features = candidate_features(gray, labels, maps)
        t1 = time.perf_counter()
        crops, sides = candidate_crops(gray, labels, maps['decode'], maps['single'])
        cnn_u = self.verifier.cnn_scores(crops, sides)
        u = self.verifier.gbm_scores(features, cnn_u)
        keep = self.verifier.keep(u)
        t2 = time.perf_counter()
        state = run_steps(InstanceState(labels, keep, maps['decode'][0], gray), self.steps)
        rles = encode_label_image(state.labels)
        t3 = time.perf_counter()
        return FrameResult(key, rles, state.labels, labels, features, cnn_u, u, keep,
                           {'candidates_features': t1-t0, 'verifier': t2-t1, 'postprocess_encode': t3-t2})

    def __call__(self, key: str, path) -> FrameResult:
        t0 = time.perf_counter()
        maps = self.predictor(path)
        t1 = time.perf_counter()
        result = self.process(key, read_gray(path), maps)
        result.seconds = {'networks': t1-t0, **result.seconds}
        result.seconds['total'] = sum(result.seconds.values())
        return result


def predict_frames(pipe: Pipeline, paths: dict[str, Path], out_dir: str | Path, log_every: int = 10) -> dict[str, dict]:
    """Run the pipeline on every frame, caching one small JSON per frame in ``out_dir`` (re-runs resume).

    Each JSON holds the final RLEs, the candidates' RLEs (decoder outlines), u, crop-CNN score,
    keep decisions and per-stage timings.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    done = {}
    computed = 0
    start = time.perf_counter()
    for i, (key, path) in enumerate(paths.items(), 1):
        record = out_dir/f'{key}.json'
        if not record.exists():
            computed += 1
            r = pipe(key, path)
            tmp = record.with_suffix('.tmp')
            tmp.write_text(json.dumps({'rles': r.rles, 'candidate_rles': encode_label_image(r.candidates),
                                       'u': r.u.tolist(), 'cnn_u': r.cnn_u.tolist(), 'keep': r.keep.tolist(),
                                       'seconds': r.seconds}))
            tmp.replace(record)
        done[key] = json.loads(record.read_text())
        if log_every and (i % log_every == 0 or i == len(paths)):
            print(f'{i}/{len(paths)} frames ({computed} computed now, {i-computed} from the cache), '
                  f'{time.perf_counter()-start:.0f} s elapsed', flush=True)
    return done
