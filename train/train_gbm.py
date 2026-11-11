"""Fit the bagged GBM verifier (weights/verifier_gbm.pkl) or, with --columns, the final pipeline's keep model.

Input per candidate: the 46 features + the out-of-fold crop-CNN score.
Target: u (mean matched IoU over annotators). Sample weight 1/(candidates in
its frame), so each frame counts once. 100 HistGradientBoosting regressors,
each on a bootstrap of frames (5 bag seeds x 20 bags); the verifier's score is
their mean. tau = 0.25 was selected on the calibration months of the
TRAIN-only round (see README) and kept frozen for this full refit.

    python train/train_gbm.py --data work/verifier_data.npz --cnn work/runs/crop_cnn/cnn_oof_scores.npz \
        --out work/runs/verifier_gbm.pkl
    # held-out evaluation model used by the notebook (fold-0 frames excluded):
    python train/train_gbm.py ... --exclude-fold 0 --out work/runs/verifier_gbm_without_fold0.pkl
    # keep model of the final pipeline: the same recipe on 59 columns (46 features, crop-CNN score, then the
    # out-of-fold context votes and detector columns from train/build_keep_columns.py, in that order)
    python train/train_gbm.py --data work/verifier_data_fz.npz --cnn work/runs/crop_cnn_fz/cnn_oof_scores.npz \
        --columns l2=work/keep_columns/context_votes.npz dv=work/keep_columns/detector_columns_y4.npz --out work/runs/keep_gbm_y4.pkl
Each --columns file holds keys, counts (same frames and candidate order as --data) and values (n_candidates, k).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

PARAMS = dict(max_iter=300, learning_rate=.04, max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=5.,
              max_bins=128, early_stopping=False)
BAG_SEEDS = [1234, 2234, 3234, 4234, 5234]
BAGS_PER_SEED = 20
TAU = .25


def frame_rows(keys, counts, features, targets, cnn, extra=()):
    """[(key, X (n, 47 + extra columns), u (n,))] per frame, frames in sorted key order."""
    rows, i = {}, 0
    for k, n in zip(keys, counts):
        rows[str(k)] = (np.hstack([features[i:i+n], cnn[i:i+n, None]]+[e[i:i+n] for e in extra]), targets[i:i+n, 1])
        i += n
    return [(k, *rows[k]) for k in sorted(rows)]


def fit_bag(rows, bags, seed, threads=8):
    """Bootstrap over frames; each bag is one HistGradientBoostingRegressor."""
    rng = np.random.default_rng(seed)
    models = []
    with threadpool_limits(limits=threads):
        for b in range(bags):
            pick = rng.integers(0, len(rows), len(rows))
            x = np.concatenate([rows[i][1] for i in pick])
            y = np.concatenate([rows[i][2] for i in pick])
            w = np.concatenate([np.full(len(rows[i][1]), 1/max(1, len(rows[i][1]))) for i in pick])
            m = HistGradientBoostingRegressor(**PARAMS, random_state=int(seed+b))
            m.fit(x, y, sample_weight=w)
            models.append(m)
    return models


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--cnn', type=Path, required=True)
    ap.add_argument('--folds', type=Path, default=Path('configs/folds_full.json'))
    ap.add_argument('--exclude-fold', type=int, help='leave this fold\'s frames out (for an honest held-out check)')
    ap.add_argument('--columns', nargs='*', default=[], metavar='NAME=NPZ',
                    help='keep model: extra column groups appended in this order (final pipeline: l2=... dv=...)')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--threads', type=int, default=8, help='OpenMP threads per fit (results do not depend on it)')
    ap.add_argument('--smoke', action='store_true', help='2 bags')
    a = ap.parse_args()
    d, c = np.load(a.data), np.load(a.cnn)
    assert (d['keys'] == c['keys']).all() and (d['counts'] == c['counts']).all()
    extra, names = [], []
    for spec in a.columns:
        name, path = spec.split('=', 1)
        e = np.load(path)
        assert (e['keys'] == d['keys']).all() and (e['counts'] == d['counts']).all(), f'{name}: rows differ from --data'
        extra.append(np.asarray(e['values'], np.float64).reshape(len(d['features']), -1))
        names.append(name)
    rows = frame_rows(d['keys'], d['counts'], d['features'], d['targets'], c['oof_u'], extra)
    if a.exclude_fold is not None:
        held = set(json.loads(a.folds.read_text())['folds'][a.exclude_fold]['heldout_sources'])
        rows = [r for r in rows if r[0] not in held]
    seeds, bags = (BAG_SEEDS[:1], 2) if a.smoke else (BAG_SEEDS, BAGS_PER_SEED)
    models = [m for seed in seeds for m in fit_bag(rows, bags, seed, a.threads)]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    bundle = {'models': models, 'cols': list(range(d['features'].shape[1])), 'tau': TAU, 'params': PARAMS,
              'n_frames': len(rows), 'n_candidates': int(sum(len(r[1]) for r in rows)), 'excluded_fold': a.exclude_fold}
    if names:  # keep model: filament_seg.keep.KeepModel reads the column groups from feature_order
        bundle.update(feature_order=['x46', 'cnn']+names, seeds=list(seeds), bags_per_seed=bags)
    with open(a.out, 'wb') as f:
        pickle.dump(bundle, f)
    print(f'{len(models)} GBMs on {len(rows)} frames', flush=True)


if __name__ == '__main__':
    main()
