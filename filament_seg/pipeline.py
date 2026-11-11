"""End-to-end frame pipeline: networks -> candidates -> verifier -> post-processing -> RLE.

FZ and lite pipelines:

    image --(EnsemblePredictor)--> decode / single / vote maps (2048^2; FZ: fused with the specialists)
          --(candidate_labels)----> candidate instances (fg >= 0.4, area >= 150)
          --(features + crop CNN + GBM)--> u per candidate, keep = u >= tau
          --(postprocess steps)---> expected-IoU outlines, rejected removed
          --(encode_label_image)--> one RLE row per kept instance

Final pipeline (``configs/final.yaml``), the same up to the FZ verifier score u_FZ, then:

          --(ContextVoter)--------> 6 context-network votes per candidate
          --(outline_nz1)---------> nz1 outlines of all candidates
          --(DetectorVoter)-------> 6 detector columns per candidate, measured on those outlines
          --(KeepModel)-----------> u from 59 columns, keep = u >= tau
          --(Refiner)-------------> refined probability of every candidate with u_FZ >= 0.10
          --(refine_pasteback)----> shifted re-drawn / re-cut masks of the kept candidates, disjoint
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
from .data import normalize_image, read_gray
from .features import candidate_features
from .inference import EnsemblePredictor
from .models import load_model, read_manifest
from .pasteback import paste_back, shifted_outlines
from .postprocess import InstanceState, apply_steps, run_steps
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
    u: np.ndarray                  # (n,) verifier score (final pipeline: keep-model score)
    keep: np.ndarray               # (n,) bool
    seconds: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)   # final pipeline: u_fz, l2 and detector columns, re-drawn flags


class Pipeline:
    """A configured pipeline (final, FZ_nz1 or lite). Model lists can be overridden (the held-out evaluation uses one fold).

    ``final_models`` overrides the final pipeline's extra models: {'context': [...], 'detectors': [...],
    'detector_groups': [...], 'keep_gbm': name, 'refiners': [...]}.
    """

    def __init__(self, cfg: dict, segmenters=None, voters=None, crop_cnns=None, gbm=None, specialists=None,
                 final_models=None):
        weights_dir = resolve(cfg['weights']['dir'])
        self.predictor = EnsemblePredictor.from_weights(cfg, weights_dir, segmenters, voters, specialists)
        self.verifier = Verifier(cfg, weights_dir, self.predictor.device, crop_cnns, gbm)
        self.settings = CandidateSettings.from_config(cfg)
        self.steps = cfg['postprocess']['steps']
        self.final = 'keep_gbm' in cfg['weights']
        if self.final:
            self._init_final(cfg, weights_dir, final_models or {})

    def _init_final(self, cfg: dict, weights_dir, over: dict):
        """Context networks, detectors, keep model and refiners of the final pipeline."""
        from .context import ContextVoter
        from .detector import DetectorVoter
        from .keep import KeepModel
        from .refiner import Refiner
        w, inf, dev = dict(cfg['weights']), dict(cfg['inference']), self.predictor.device
        w.update({k: v for k, v in over.items() if k in ('context', 'detectors', 'keep_gbm', 'refiners')})
        if 'detector_groups' in over:
            inf['detector'] = {**inf['detector'], 'groups': over['detector_groups']}
        cfg = {**cfg, 'weights': w, 'inference': inf}
        assert self.steps[0]['name'] == 'outline_nz1' and self.steps[-1]['name'] == 'refine_pasteback'
        ctx = inf['context']
        self.context = ContextVoter([load_model(n, weights_dir, dev) for n in w['context']], ctx['view_batch'], ctx['autocast'])
        self.detector = DetectorVoter(cfg, weights_dir, read_manifest(weights_dir), device=dev)
        self.keep_model = KeepModel(resolve(w['dir'])/w['keep_gbm'], cfg['verifier']['tau'])
        r = inf['refiner']
        self.refiner = Refiner([load_model(n, weights_dir, dev) for n in w['refiners']], dev, r.get('batch'),
                               r['autocast'], r['flips'], r['u_min'])

    def process(self, key: str, gray: np.ndarray, maps: dict[str, np.ndarray]) -> FrameResult:
        """Everything after the networks, for one frame."""
        if self.final:
            return self.process_final(key, gray, maps)
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

    def process_final(self, key: str, gray: np.ndarray, maps: dict[str, np.ndarray], detections=None,
                      context_map=None, refined=None, u_fz=None) -> FrameResult:
        """Final pipeline after the dense networks. ``maps`` must hold the per-fold maps (``keep_folds=True``).

        ``detections`` (one instance set per detector), ``context_map``, ``refined`` and ``u_fz`` may be passed in to
        replay stored intermediate results (verification); by default they are computed here.
        """
        import torch
        sec = {}
        t = time.perf_counter()
        labels = candidate_labels(maps['decode'][0], self.settings)
        n = int(labels.max())
        features = candidate_features(gray, labels, maps)
        crops, sides = candidate_crops(gray, labels, maps['decode'], maps['single'])
        cnn_u = self.verifier.cnn_scores(crops, sides)
        if u_fz is None:
            u_fz = self.verifier.gbm_scores(features, cnn_u)
        sec['candidates_verifier'], t = time.perf_counter()-t, time.perf_counter()
        if context_map is None:
            x = torch.from_numpy(maps['input'])[None, None].to(self.predictor.device)
            context_map = self.context.maps(x, maps['cnx_folds'], maps['r34_folds'])
        l2 = self.context.votes(labels, context_map)
        sec['context'], t = time.perf_counter()-t, time.perf_counter()
        fg = maps['decode'][0]
        seeds = apply_steps(InstanceState(labels, np.ones(n, bool), fg, gray), self.steps[:1])
        choice = seeds.choice if seeds.choice is not None else np.zeros(0, int)
        dv = self.detector.votes(gray, seeds.labels, n, detections)
        sec['detectors'], t = time.perf_counter()-t, time.perf_counter()
        u = self.keep_model.scores({'x46': features, 'cnn': cnn_u, 'l2': l2, 'dv': dv})
        keep = self.keep_model.keep(u)
        if refined is None:
            refined = self.refiner.refine(normalize_image(gray), seeds.labels, u_fz, maps['fused_folds'])
        sec['refiner'], t = time.perf_counter()-t, time.perf_counter()
        p = {k: v for k, v in self.steps[-1].items() if k != 'name'}
        shift = p.get('shift')
        nz = self.steps[0]
        seeds_shifted = (shifted_outlines(labels, fg, choice, shift, nz['levels'], nz['base_level'], nz['grow_px'])
                         if shift is not None and n else None)
        final, cand_ids, redrawn = paste_back(seeds.labels, keep, u_fz, refined, seeds_shifted, p['thr'], p['fill'],
                                              p['umax'], shift)
        rles = encode_label_image(final)
        # FZ_nz1's own output comes for free: the nz1 outlines of the candidates with u_FZ >= tau
        fz_keep = np.r_[0, np.cumsum(u_fz >= self.verifier.tau)*(u_fz >= self.verifier.tau)].astype(np.int32)
        fz_rles = encode_label_image(fz_keep[seeds.labels])
        sec['pasteback_encode'] = time.perf_counter()-t
        return FrameResult(key, rles, final, labels, features, cnn_u, u, keep, sec,
                           {'u_fz': u_fz, 'l2': l2, 'dv': dv, 'candidates_of_masks': cand_ids,
                            'redrawn': [int(redrawn[j]) for j in cand_ids], 'refined': sorted(refined), 'fz_rles': fz_rles})

    def __call__(self, key: str, path) -> FrameResult:
        t0 = time.perf_counter()
        maps = self.predictor(path, keep_folds=self.final)
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
            record_data = {'rles': r.rles, 'candidate_rles': encode_label_image(r.candidates),
                           'u': r.u.tolist(), 'cnn_u': r.cnn_u.tolist(), 'keep': r.keep.tolist(), 'seconds': r.seconds}
            if r.extra:
                record_data.update({k: np.asarray(v).tolist() for k, v in r.extra.items()})
            tmp.write_text(json.dumps(record_data))
            tmp.replace(record)
        done[key] = json.loads(record.read_text())
        if log_every and (i % log_every == 0 or i == len(paths)):
            print(f'{i}/{len(paths)} frames ({computed} computed now, {i-computed} from the cache), '
                  f'{time.perf_counter()-start:.0f} s elapsed', flush=True)
    return done
