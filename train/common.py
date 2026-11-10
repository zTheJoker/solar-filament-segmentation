"""Shared training pieces: calendar-month folds, datasets, losses, small utilities."""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

FOLDS, PURGE_DAYS, INNER_FRACTION, SEED = 5, 4, .1, 20261108


def day_ordinal(sid: str) -> int:
    return datetime.strptime(sid[:8], '%Y%m%d').date().toordinal()


def make_folds(sources: list[str]) -> dict:
    """5 outer folds of whole calendar months (greedy-balanced by frame count).

    Training frames within PURGE_DAYS of a held-out frame are purged from that
    fold (filaments persist for days); about 10% of the remaining frames, as
    whole months, form the inner slice used for checkpoint selection.
    """
    sources = sorted(set(sources))
    months = sorted({s[:6] for s in sources})
    size = {m: sum(s[:6] == m for s in sources) for m in months}
    order = months[:]
    random.Random(SEED).shuffle(order)
    load, assign = [0]*FOLDS, {}
    for m in sorted(order, key=lambda m: -size[m]):
        k = min(range(FOLDS), key=lambda i: (load[i], i))
        assign[m] = k
        load[k] += size[m]
    folds = []
    for k in range(FOLDS):
        held = [s for s in sources if assign[s[:6]] == k]
        held_days = np.asarray(sorted({day_ordinal(s) for s in held}))
        rest = [s for s in sources if assign[s[:6]] != k]
        purged = [s for s in rest if np.min(np.abs(held_days-day_ordinal(s))) <= PURGE_DAYS]
        kept = [s for s in rest if s not in set(purged)]
        shuffled = sorted({s[:6] for s in kept})
        random.Random(SEED+k).shuffle(shuffled)
        inner_months, count = set(), 0
        for m in shuffled:
            if count >= INNER_FRACTION*len(kept):
                break
            inner_months.add(m)
            count += sum(s[:6] == m for s in kept)
        folds.append({'fold': k, 'heldout_sources': held,
                      'train_sources': [s for s in kept if s[:6] not in inner_months],
                      'inner_validation_sources': [s for s in kept if s[:6] in inner_months],
                      'purged_sources': purged, 'heldout_months': sorted({s[:6] for s in held}),
                      'inner_validation_months': sorted(inner_months)})
    return {'folds': folds, 'train_sources': sources, 'purge_days': PURGE_DAYS, 'inner_fraction': INNER_FRACTION,
            'seed': SEED}


def load_fold(path: Path, fold: int) -> dict:
    return json.loads(Path(path).read_text())['folds'][fold]


def records_by_source(cache: Path, keep: set[str]) -> dict[str, list[dict]]:
    out = {}
    for r in json.loads((Path(cache)/'manifest.json').read_text())['records']:
        if r['source_id'] in keep:
            out.setdefault(r['source_id'], []).append(r)
    return out


def dihedral(x, y):
    """Random rot90 + horizontal flip of an image (H, W) and its target (C, H, W)."""
    k = random.randrange(4)
    x, y = np.rot90(x, k), np.rot90(y, k, axes=(-2, -1))
    if random.random() < .5:
        x, y = x[:, ::-1], y[:, :, ::-1]
    return x, y


def photometric(x):
    """Random gamma (p = .7) and gain/offset jitter in normalised units."""
    if random.random() < .7:
        x = (np.clip(x*.25+.5, 0, 1)**random.uniform(.75, 1.35)-.5)/.25
    return x*random.uniform(.9, 1.1)+random.uniform(-.1, .1)


class SourceMeanDataset(Dataset):
    """ConvNeXt recipe: one item per frame; target = mean over its annotators (soft), plus shift-scale-rotate."""

    def __init__(self, cache: Path, groups: dict[str, list[dict]], augment: bool):
        self.cache, self.groups, self.augment = Path(cache), groups, augment
        self.keys = sorted(groups)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        import cv2
        sid = self.keys[i]
        x = np.load(self.cache/'images'/f'{sid}.npy').astype(np.float32)
        masks, edges = [], []
        for r in self.groups[sid]:
            with np.load(self.cache/r['cache_target']) as t:
                masks.append(t['mask'].astype(np.float32))
                edges.append(t['boundary'].astype(np.float32))
        y = np.stack([np.mean(masks, 0), np.mean(edges, 0)])
        if self.augment:
            x, y = dihedral(x, y)
            x, y = np.ascontiguousarray(x), np.ascontiguousarray(y)
            if random.random() < .7:  # shift-scale-rotate
                h, w = x.shape
                m = cv2.getRotationMatrix2D((w/2-.5, h/2-.5), random.uniform(-30, 30), random.uniform(.8, 1.2))
                m[:, 2] += np.random.uniform(-.0625, .0625, 2)*w
                warp = lambda a: cv2.warpAffine(a, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
                x = warp(x)
                y = np.stack([warp(y[0]), warp(y[1])])
            x = photometric(x)
        return torch.from_numpy(np.ascontiguousarray(x[None])).float(), torch.from_numpy(np.ascontiguousarray(y)).float()


class RecordDataset(Dataset):
    """ResNet34 recipe: one item per annotator record (hard union target), dihedral + both flips."""

    def __init__(self, cache: Path, records: list[dict], augment: bool):
        self.cache, self.rows, self.augment = Path(cache), records, augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        x = np.load(self.cache/'images'/f"{r['source_id']}.npy")
        with np.load(self.cache/r['cache_target']) as t:
            y = np.stack([t['mask'].astype(np.float32), t['boundary'].astype(np.float32)])
        if self.augment:
            x, y = dihedral(x, y)
            if random.random() < .5:
                x, y = x[::-1], y[:, ::-1, :]
            x = photometric(x)
        return torch.from_numpy(np.ascontiguousarray(x[None])).float(), torch.from_numpy(np.ascontiguousarray(y)).float()


def segmentation_loss(logits, target, pos_weight):
    """.5 BCE(fg, pos_weight) + .5 soft Dice(fg) + .15 BCE(boundary, pos_weight 5)."""
    fg, edge_logit = logits[:, 0:1].float(), logits[:, 1:2].float()
    truth, edge = target[:, 0:1], target[:, 1:2]
    bce = F.binary_cross_entropy_with_logits(fg, truth, pos_weight=fg.new_tensor(pos_weight))
    p = fg.sigmoid()
    dims = (1, 2, 3)
    dice = 1-((2*(p*truth).sum(dims)+1)/((p+truth).sum(dims)+1)).mean()
    edge_loss = F.binary_cross_entropy_with_logits(edge_logit, edge, pos_weight=fg.new_tensor(5.))
    return .5*bce+.5*dice+.15*edge_loss


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device_and_autocast():
    """CUDA with bf16 autocast for real runs; CPU (bf16 autocast) only for --smoke checks."""
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return dev, (lambda: torch.autocast(device_type=dev.type, dtype=torch.bfloat16))


def atomic_save(value, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    torch.save(value, tmp)
    tmp.replace(path)
