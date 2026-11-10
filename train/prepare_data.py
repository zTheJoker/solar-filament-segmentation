"""Build the training caches from the Kaggle data (run once, CPU).

For every labelled frame and model side S (768 for the ResNet34 first stage,
1024 for everything else):
  images/<source>.npy      normalised network input (filament_seg.data.model_input)
  targets/<record>.npz     per annotator record: 'mask' = BOX-resampled union
                           occupancy of that annotator's filaments (soft, fp16),
                           'boundary' = 1-px instance contours at side S
manifest.json lists every annotator record with its source id.

    python train/prepare_data.py --out work/prepared --sides 768 1024 --workers 8
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from filament_seg.config import data_path, load_config  # noqa: E402
from filament_seg.data import list_images, model_input, source_id  # noqa: E402


def decode(segmentation, height, width) -> np.ndarray:
    from pycocotools import mask as cm
    if isinstance(segmentation, dict):
        rle = segmentation if not isinstance(segmentation.get('counts'), list) else cm.frPyObjects(segmentation, height, width)
    else:
        polygons = segmentation if isinstance(segmentation[0], list) else [segmentation]
        rle = cm.merge(cm.frPyObjects(polygons, height, width))
    m = cm.decode(rle)
    return (m.any(axis=2) if m.ndim == 3 else m).astype(bool)


def record_target(job):
    """Soft occupancy + contour target of one annotator record at one side."""
    image, anns, side, out = job
    h, w = int(image['height']), int(image['width'])
    union = np.zeros((h, w), bool)
    boundary = np.zeros((side, side), bool)
    for ann in sorted(anns, key=lambda a: -float(a.get('area', 0))):
        m = decode(ann['segmentation'], h, w)
        if not m.any():
            continue
        union |= m
        small = np.asarray(Image.fromarray(m.astype(np.uint8)*255).resize((side, side), Image.Resampling.BOX)) > 0
        boundary |= binary_dilation(small) ^ binary_erosion(small)
    occupancy = np.asarray(Image.fromarray(union.astype(np.uint8)*255).resize((side, side), Image.Resampling.BOX),
                           np.float32)/255
    name = f"targets/{hashlib.sha256(str(image['id']).encode()).hexdigest()[:24]}.npz"
    np.savez_compressed(Path(out)/name, mask=occupancy.astype(np.float16), boundary=boundary.astype(np.uint8))
    return {'annotation_image_id': str(image['id']), 'source_id': source_id(image['file_name']), 'cache_target': name}


def image_input(job):
    sid, path, side, out = job
    np.save(Path(out)/'images'/f'{sid}.npy', model_input(path, side), allow_pickle=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default='configs/f_full_refit_nz1.yaml')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--sides', type=int, nargs='+', default=[768, 1024])
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--smoke', action='store_true', help='only 8 frames spread over the date range')
    a = ap.parse_args()
    cfg = load_config(a.config)
    coco = json.loads(data_path(cfg, 'train_annotations').read_text())
    paths = list_images(data_path(cfg, 'train_images'))
    anns = defaultdict(list)
    for ann in coco['annotations']:
        anns[str(ann['image_id'])].append(ann)
    images = coco['images']
    if a.smoke:
        keep = sorted(paths)[::len(paths)//8][:8]
        images = [im for im in images if source_id(im['file_name']) in keep]
        paths = {k: paths[k] for k in keep}
    for side in a.sides:
        out = a.out/str(side)
        (out/'images').mkdir(parents=True, exist_ok=True)
        (out/'targets').mkdir(parents=True, exist_ok=True)
        with ProcessPoolExecutor(a.workers) as pool:
            list(pool.map(image_input, [(k, str(p), side, str(out)) for k, p in paths.items()]))
            records = list(pool.map(record_target, [(im, anns[str(im['id'])], side, str(out)) for im in images]))
        (out/'manifest.json').write_text(json.dumps({'side': side, 'records': records}, indent=1))
        print(f'side {side}: {len(paths)} frames, {len(records)} annotator records', flush=True)


if __name__ == '__main__':
    main()
