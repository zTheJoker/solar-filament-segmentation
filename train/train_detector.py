"""YOLO26m-seg detectors of the final pipeline (weights/yolo_s{seed}_f{fold}.pt): build a fold's dataset, train it.

The detectors are only used through the six agreement columns of the keep model (filament_seg/detector.py).

Data: one training sample per annotator record (a frame with three annotators appears three times, each time with
its own drawing). Each annotated instance is decoded with pycocotools, exactly as the host's scorer rasterises it,
and turned into its outer contour; an instance with several pieces keeps every piece of at least 4 px as one
polygon joined by zero-width bridges, so the raster the detector learns is the scorer's raster. Images are the
native 8-bit grayscale frames saved as PNG. Fold k trains on its ``train_sources`` and selects its checkpoint on its
``inner_validation_sources`` (configs/folds_full.json); its held-out frames are never seen.

Recipe (Ultralytics 8.4.210): official COCO ``yolo26m-seg.pt`` weights, one class, imgsz 1536, 80 epochs, batch 4,
optimizer auto (AdamW, lr .002), mosaic with close_mosaic 10, flips on both axes, value jitter .3 and no hue or
saturation jitter, amp, best epoch by Ultralytics fitness on the inner slice. Seeds 0-3. About 32-41 minutes per fold
on one RTX 5090. Telemetry and every logger integration are switched off before training.

    python train/train_detector.py data --fold 0 --out work/detector                         # PNGs + YOLO labels
    python train/train_detector.py train --fold 0 --seed 0 --out work/detector \
        --weights-in work/yolo26m-seg.pt                                                     # -> work/detector/runs/s0/f0
The COCO starting weights come from the Ultralytics release (``yolo26m-seg.pt``, AGPL-3.0); download them once.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ['WANDB_MODE'] = 'disabled'
os.environ['YOLO_OFFLINE'] = '1'
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from filament_seg.config import data_path, load_config  # noqa: E402
from filament_seg.data import list_images, load_ground_truth, read_gray  # noqa: E402


def polygon_of(mask: np.ndarray):
    """Outer contour(s) of a binary mask as one (N, 2) float32 polygon in pixels; pieces joined by zero-width bridges."""
    import cv2
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cs = [c[:, 0, :].astype(np.float32) for c in cs if len(c) >= 3 and cv2.contourArea(c) >= 4]
    if not cs:
        return None
    cs.sort(key=lambda c: -len(c))
    out = cs[0]
    for c in cs[1:]:  # walk piece 0, jump to the nearest point of the next piece, walk it fully, jump back
        d = ((out[:, None, :]-c[None, :, :])**2).sum(-1)
        i, j = np.unravel_index(np.argmin(d), d.shape)
        c2 = np.roll(c, -j, 0)
        out = np.concatenate([out[:i+1], c2, c2[:1], out[i:]], 0)
    return out


def cmd_data(a):
    from PIL import Image
    from pycocotools import mask as cm
    cfg = load_config(a.config)
    fold = json.loads(a.folds.read_text())['folds'][a.fold]
    tr, iv = sorted(fold['train_sources']), sorted(fold['inner_validation_sources'])
    assert not (set(tr) | set(iv)) & (set(fold['heldout_sources']) | set(fold.get('purged_sources', []))), 'cross-fitting violated'
    if a.smoke:
        tr, iv = tr[:4], iv[:2]
    paths = list_images(data_path(cfg, 'train_images'))
    gt = load_ground_truth(data_path(cfg, 'train_annotations'), set(tr) | set(iv))
    png = a.out/'png'
    png.mkdir(parents=True, exist_ok=True)
    root = a.out/'datasets'/f'f{a.fold}'
    counts = {}
    for split, keys in (('train', tr), ('val', iv)):
        (root/'images'/split).mkdir(parents=True, exist_ok=True)
        (root/'labels'/split).mkdir(parents=True, exist_ok=True)
        n_rec = n_inst = 0
        for key in keys:
            if not (png/f'{key}.png').exists():
                Image.fromarray(read_gray(paths[key])).save(png/f'{key}.png')
            for i, (_, rles) in enumerate(gt[key]):
                stem = f'{key}_r{i}'
                img = root/'images'/split/f'{stem}.png'
                if not img.exists():
                    os.symlink((png/f'{key}.png').resolve(), img)
                lines = []
                for r in rles:
                    p = polygon_of(cm.decode(r).astype(bool))
                    if p is not None:
                        lines.append('0 '+' '.join(f'{v:.6f}' for v in np.clip(p/2048., 0, 1).reshape(-1)))
                (root/'labels'/split/f'{stem}.txt').write_text('\n'.join(lines)+('\n' if lines else ''))
                n_rec += 1
                n_inst += len(lines)
        counts[split] = {'frames': len(keys), 'records': n_rec, 'instances': n_inst}
    (root/'data.yaml').write_text(f'path: {root.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: filament\n')
    (root/'lists.json').write_text(json.dumps({'fold': a.fold, 'counts': counts, 'train_sources': tr, 'inner_validation_sources': iv}))
    print('DATA', a.fold, json.dumps(counts), flush=True)


def cmd_train(a):
    from ultralytics import YOLO, settings
    settings.update({k: False for k in ('sync', 'wandb', 'mlflow', 'comet', 'clearml', 'neptune', 'dvc', 'tensorboard',
                                        'raytune', 'hub') if k in settings})
    root = a.out/'datasets'/f'f{a.fold}'
    out = a.out/'runs'/f's{a.seed}'
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    YOLO(str(a.weights_in)).train(
        data=str(root/'data.yaml'), imgsz=a.imgsz, epochs=1 if a.smoke else a.epochs, batch=a.batch, workers=a.workers,
        seed=a.seed, deterministic=False, project=str(out), name=f'f{a.fold}', exist_ok=True, single_cls=True, cache=a.cache,
        fliplr=.5, flipud=.5, hsv_h=0., hsv_s=0., hsv_v=.3, close_mosaic=10, plots=False, amp=True, patience=1000,
        optimizer='auto', verbose=False)
    print('TRAINED fold', a.fold, 'seed', a.seed, f'{(time.time()-t0)/60:.1f} min ->', out/f'f{a.fold}'/'weights'/'best.pt', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['data', 'train'])
    ap.add_argument('--config', default='configs/final.yaml')
    ap.add_argument('--folds', type=Path, default=Path('configs/folds_full.json'))
    ap.add_argument('--fold', type=int, required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--weights-in', type=Path, default=Path('work/yolo26m-seg.pt'), help='COCO starting weights')
    ap.add_argument('--imgsz', type=int, default=1536)
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--cache', default='ram')
    ap.add_argument('--smoke', action='store_true', help='data: 4 + 2 frames; train: 1 epoch')
    a = ap.parse_args()
    {'data': cmd_data, 'train': cmd_train}[a.cmd](a)


if __name__ == '__main__':
    main()
