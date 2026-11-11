"""Out-of-fold context votes and detector columns of every labelled frame (the extra inputs of the keep model).

Every labelled frame of fold k is processed only by fold-k networks, which never saw it: ConvNeXt k + specialist k
(fused map -> candidates, the same candidates as build_verifier_data.py), ConvNeXt k + ResNet34 k -> context network k
-> six vote features, and the fold-k detector of each seed -> six columns on the nz1 outlines, averaged over the
seeds (one model per seed, so the columns of a training frame come from four detectors, those of a test frame from
twenty). Rows are in the order of build_verifier_data.py's table, so train_gbm.py --columns can append them.

    python train/build_keep_columns.py --data work/verifier_data_fz.npz --out work/keep_columns          # released weights
    python train/build_keep_columns.py ... --seeds 0                                                     # single-seed sibling
-> context_votes.npz, detector_columns_<tag>.npz (keys, counts, values)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from filament_seg.candidates import CandidateSettings, candidate_labels  # noqa: E402
from filament_seg.config import data_path, load_config, resolve  # noqa: E402
from filament_seg.context import ContextVoter  # noqa: E402
from filament_seg.data import list_images, read_gray  # noqa: E402
from filament_seg.detector import DetectorVoter  # noqa: E402
from filament_seg.inference import EnsemblePredictor  # noqa: E402
from filament_seg.models import load_model, read_manifest  # noqa: E402
from filament_seg.postprocess import InstanceState, apply_steps  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default='configs/final.yaml')
    ap.add_argument('--data', type=Path, required=True, help='build_verifier_data.py output (defines frames and row order)')
    ap.add_argument('--seeds', default='0,1,2,3')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--smoke', action='store_true', help='2 frames per fold')
    a = ap.parse_args()
    cfg = load_config(a.config)
    wd = resolve(cfg['weights']['dir'])
    seeds = [int(s) for s in a.seeds.split(',')]
    folds = json.loads(resolve(cfg['data']['folds']).read_text())['folds']
    paths = list_images(data_path(cfg, 'train_images'))
    table = np.load(a.data)
    counts = dict(zip(table['keys'].tolist(), table['counts'].tolist()))
    settings, steps = CandidateSettings.from_config(cfg), cfg['postprocess']['steps'][:1]
    ic = cfg['inference']['context']
    l2, dv = {}, {}
    for f in folds:
        k = f['fold']
        pred = EnsemblePredictor.from_weights(cfg, wd, [f'cnx_fold{k}'], [f'r34_fold{k}'], [f'spec_fold{k}'])
        context = ContextVoter([load_model(f'l2_fold{k}', wd, pred.device)], ic['view_batch'], ic['autocast'])
        det_cfg = {**cfg, 'inference': {**cfg['inference'], 'detector': {**cfg['inference']['detector'], 'groups': [1]*len(seeds)}}}
        detector = DetectorVoter(det_cfg, wd, read_manifest(wd), [f'yolo_s{s}_f{k}' for s in seeds], pred.device)
        for key in sorted(f['heldout_sources'])[:2 if a.smoke else None]:
            maps = pred(paths[key], keep_folds=True)
            gray = read_gray(paths[key])
            labels = candidate_labels(maps['decode'][0], settings)
            n = int(labels.max())
            assert n == counts[key], f'{key}: {n} candidates, the verifier table has {counts[key]}'
            x = torch.from_numpy(maps['input'])[None, None].to(pred.device)
            l2[key] = context.votes(labels, context.maps(x, maps['cnx_folds'], maps['r34_folds']))
            seeds_img = apply_steps(InstanceState(labels, np.ones(n, bool), maps['decode'][0], gray), steps).labels
            dv[key] = detector.votes(gray, seeds_img, n)
        print('fold', k, 'done', flush=True)
    keys = [k for k in table['keys'].tolist() if k in l2]
    a.out.mkdir(parents=True, exist_ok=True)
    tag = 'y4' if len(seeds) == 4 else 's'+''.join(map(str, seeds))
    for name, cols in (('context_votes', l2), (f'detector_columns_{tag}', dv)):
        np.savez(a.out/f'{name}.npz', keys=np.asarray(keys), counts=np.asarray([counts[k] for k in keys]),
                 values=np.concatenate([np.asarray(cols[k], np.float64).reshape(-1, 6) for k in keys]))
    print('wrote', a.out, len(keys), 'frames', flush=True)


if __name__ == '__main__':
    main()
