"""Image discovery, preprocessing and ground-truth loading.

Preprocessing uses only the image itself (no dataset statistics, no metadata):
the 2048x2048 H-alpha frame is converted to 8-bit grayscale, resized to the
model side with PIL bilinear resampling, then linearly rescaled so that the
1st/99th intensity percentiles inside a central disk map to 0/1, clipped, and
centred to (v - 0.5) / 0.25.
"""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image

NATIVE = 2048
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png'}


def source_id(name: str | Path) -> str:
    """Physical observation id ``YYYYMMDDhhmmssXh`` contained in a file name or annotator-image id."""
    matches = re.findall(r'\d{14}[A-Za-z]{2}', str(name))
    if not matches:
        raise ValueError(f'Unrecognized observation name: {name!r}')
    return matches[-1]


def list_images(directory: str | Path) -> dict[str, Path]:
    """Map source id -> image path for every image file below ``directory`` (sorted by id)."""
    paths = {}
    for path in Path(directory).rglob('*'):
        if path.suffix.lower() in IMAGE_SUFFIXES:
            key = source_id(path.name)
            if key in paths:
                raise ValueError(f'Duplicate observation {key}')
            paths[key] = path
    return dict(sorted(paths.items()))


def read_gray(path: str | Path) -> np.ndarray:
    """Native 8-bit grayscale frame (2048 x 2048, uint8)."""
    with Image.open(path) as image:
        gray = np.asarray(image.convert('L'))
    if gray.shape != (NATIVE, NATIVE):
        raise ValueError(f'{path}: expected a native {NATIVE}x{NATIVE} frame, got {gray.shape}')
    return gray


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Per-image percentile scaling: 1st/99th percentile inside a 0.45-radius disk -> [0, 1], centred, /0.25."""
    image = np.asarray(image, dtype=np.float32)
    h, w = image.shape
    yy, xx = np.ogrid[:h, :w]
    disk = ((yy-(h-1)/2)/(h*.45))**2+((xx-(w-1)/2)/(w*.45))**2 < 1
    lo, hi = np.percentile(image[disk], [1, 99])
    value = np.clip((image-lo)/max(float(hi-lo), 5.), 0., 1.)
    return ((value-.5)/.25).astype(np.float32)


def model_input(path: str | Path, side: int = 1024) -> np.ndarray:
    """Network input for one frame: grayscale -> bilinear resize to ``side`` -> normalize_image."""
    with Image.open(path) as image:
        image = image.convert('L')
        if image.size != (NATIVE, NATIVE):
            raise ValueError(f'{path}: not a native {NATIVE}x{NATIVE} frame')
        small = np.asarray(image.resize((side, side), Image.Resampling.BILINEAR))
    return normalize_image(small)


def segmentation_to_rle(segmentation, shape=(NATIVE, NATIVE)) -> dict:
    """COCO polygon / RLE annotation -> compressed COCO RLE (pycocotools)."""
    from pycocotools import mask as cm
    h, w = shape
    if isinstance(segmentation, dict):
        counts = segmentation['counts']
        if isinstance(counts, list):
            return cm.frPyObjects({'size': [h, w], 'counts': counts}, h, w)
        return {'size': [h, w], 'counts': counts.encode('ascii') if isinstance(counts, str) else counts}
    polygons = segmentation if isinstance(segmentation[0], list) else [segmentation]
    return cm.merge(cm.frPyObjects(polygons, h, w))


def load_ground_truth(annotation_json: str | Path, sources: set[str] | None = None) -> dict[str, list]:
    """Per physical source: list of (annotator_image_id, [COCO RLE per filament]).

    One physical frame is labelled by several annotators; every annotator-image
    record is kept separate, exactly as in the host's evaluation.
    """
    coco = json.loads(Path(annotation_json).read_text())
    by_image = defaultdict(list)
    for ann in coco['annotations']:
        by_image[str(ann['image_id'])].append(ann)
    result = defaultdict(list)
    for image in coco['images']:
        key = source_id(image['file_name'])
        if sources is not None and key not in sources:
            continue
        shape = (int(image['height']), int(image['width']))
        rles = [segmentation_to_rle(a['segmentation'], shape) for a in by_image[str(image['id'])]]
        result[key].append((str(image['id']), rles))
    return dict(result)
