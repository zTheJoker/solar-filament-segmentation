"""Out-of-fold candidate table for the verifier (features, targets, crops).

Every labelled frame is predicted only by the fold networks that did NOT see
it in training (ConvNeXt fold k for the maps, ResNet34 fold k for the votes,
and in FZ mode specialist fold k fused into the maps), with the same TTA,
decoding and features as inference. Each candidate gets
two targets from the annotations, averaged over the frame's annotators:
  match = 1 if one-to-one matched with IoU > .5 (Hungarian, max cardinality)
  u     = matched IoU (0 if unmatched)  -> the candidate's share of PQ
Output npz: keys, counts (candidates per frame), features (N, 46),
targets (N, 2) = [match, u], crops (N, 5, 128, 128) uint8, sides (N,).

    python train/build_verifier_data.py --config configs/f_lite.yaml --runs work/runs --out work/verifier_data_lite.npz
    python train/build_verifier_data.py --config configs/fz_nz1.yaml --runs work/runs --out work/verifier_data_fz.npz
    # --runs holds cnx_fold{k}/, r34_fold{k}/ (and spec_fold{k}/ for FZ) best.pt; --from-weights uses weights/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from filament_seg.candidates import CandidateSettings, candidate_labels  # noqa: E402
from filament_seg.config import data_path, load_config, resolve  # noqa: E402
from filament_seg.data import list_images, load_ground_truth, read_gray  # noqa: E402
from filament_seg.features import candidate_features  # noqa: E402
from filament_seg.inference import EnsemblePredictor, pick_device  # noqa: E402
from filament_seg.metrics import overlap_matrices  # noqa: E402
from filament_seg.models import ConvNeXtUNet, ResNet34UNet  # noqa: E402
from filament_seg.submission import encode_label_image  # noqa: E402
from filament_seg.verifier import candidate_crops  # noqa: E402


def match_targets(iou: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per prediction: matched (0/1) and matched IoU under strict >.5 one-to-one matching."""
    matched = np.zeros(iou.shape[1])
    matched_iou = np.zeros(iou.shape[1])
    if not iou.size:
        return matched, matched_iou
    hit = iou > .5
    rows, cols = linear_sum_assignment(-(hit*(min(iou.shape)+1)+iou*hit))
    ok = hit[rows, cols]
    matched[cols[ok]] = 1.
    matched_iou[cols[ok]] = iou[rows[ok], cols[ok]]
    return matched, matched_iou


def load_checkpoint_net(path, device):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    cfg = ck.get('config', {})
    net = ConvNeXtUNet(cfg['encoder']) if cfg.get('architecture') == 'ConvNeXtUNet' else ResNet34UNet()
    net.load_state_dict(ck['model'])
    return net.to(device).eval()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default='configs/fz_nz1.yaml', help='fz_nz1.yaml (fused maps) or f_lite.yaml')
    ap.add_argument('--runs', type=Path, help="folder with cnx_fold{k}/, r34_fold{k}/ (FZ: also spec_fold{k}/) best.pt")
    ap.add_argument('--from-weights', action='store_true', help='use the released weights/ instead of --runs')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--smoke', action='store_true', help='2 frames per fold')
    a = ap.parse_args()
    cfg = load_config(a.config)
    device = pick_device(cfg['inference']['device'])
    folds = json.loads(resolve(cfg['data']['folds']).read_text())['folds']
    paths = list_images(data_path(cfg, 'train_images'))
    settings = CandidateSettings.from_config(cfg)
    out = {k: [] for k in ['keys', 'counts', 'features', 'targets', 'crops', 'sides']}
    for fold in folds:
        k = fold['fold']
        fz = bool(cfg['weights'].get('specialists'))
        if a.from_weights:
            pred = EnsemblePredictor.from_weights(cfg, resolve(cfg['weights']['dir']), [f'cnx_fold{k}'], [f'r34_fold{k}'],
                                                  [f'spec_fold{k}'] if fz else [])
        else:
            net = lambda name: load_checkpoint_net(a.runs/f'{name}_fold{k}'/'best.pt', device)
            pred = EnsemblePredictor(cfg, [net('cnx')], [net('r34')], [net('spec')] if fz else [], device)
        keys = sorted(fold['heldout_sources'])[:2 if a.smoke else None]
        truth = load_ground_truth(data_path(cfg, 'train_annotations'), set(keys))
        for key in keys:
            maps = pred(paths[key])
            gray = read_gray(paths[key])
            labels = candidate_labels(maps['decode'][0], settings)
            rles = [{'size': [2048, 2048], 'counts': r.encode('ascii')} for r in encode_label_image(labels)]
            per_annotator = [np.column_stack(match_targets(overlap_matrices(gt, rles)[0])) for _, gt in truth[key]]
            crops, sides = candidate_crops(gray, labels, maps['decode'], maps['single'])
            out['keys'].append(key)
            out['counts'].append(int(labels.max()))
            out['features'].append(candidate_features(gray, labels, maps))
            out['targets'].append(np.mean(per_annotator, 0) if per_annotator else np.zeros((int(labels.max()), 2)))
            out['crops'].append(crops)
            out['sides'].append(sides)
        print(f'fold {k}: {len(keys)} frames', flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, keys=np.asarray(out['keys']), counts=np.asarray(out['counts']),
             **{k: np.concatenate(out[k]) for k in ['features', 'targets', 'crops', 'sides']})


if __name__ == '__main__':
    main()
