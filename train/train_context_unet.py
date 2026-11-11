"""Context networks of the final pipeline (weights/l2_fold*.safetensors): prepare their inputs, train one fold.

Input (3 channels at 1024): the frame's network input (prepare_data.py cache) and the out-of-fold D4 foreground maps of
the ConvNeXt segmenter and the ResNet34 voter, stored as uint8 (round(255 p)) and fed as (p - 0.5) / 0.25.
Target and loss: as the ConvNeXt segmenter (annotator-mean occupancy + boundary; .5 BCE + .5 soft Dice + .15 boundary).

Honesty: every labelled frame gets its maps from the fold networks that held it out (``prep``), so context fold k is
trained on maps that, like the test maps, come from networks that never saw the frame. At test time context network k
is fed the maps of segmenter k and voter k (filament_seg/context.py).

Recipe: ConvNeXt-T U-Net (ImageNet-22k encoder, 3-channel stem), 40 epochs, batch 8, AdamW lr 2e-4 wd .01, 200-step
warm-up + cosine, grad-clip 3, bf16 autocast, EMA .999 (warm-started); the EMA weights with the lowest loss on the
fold's inner month slice are saved. Seed 71 + fold. The released networks trained in about 40 minutes per fold on one
GPU (three folds sharing it).

    python train/train_context_unet.py prep --cache work/prepared/1024 --runs work/runs --out work/context_maps
    python train/train_context_unet.py train --cache work/prepared/1024 --maps work/context_maps --fold 0 --out work/runs/l2_fold0
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import atomic_save, device_and_autocast, load_fold, photometric, records_by_source, seed_everything, segmentation_loss  # noqa: E402
from filament_seg.models import ConvNeXtUNet  # noqa: E402


def prep(a):
    """uint8 (2, 1024, 1024) out-of-fold [ConvNeXt, ResNet34] D4 foreground per labelled frame -> <out>/<source>.npy."""
    from filament_seg.inference import predict_tta
    from filament_seg.models import load_model
    folds = json.loads(a.folds.read_text())['folds']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    a.out.mkdir(parents=True, exist_ok=True)
    for f in folds:
        k = f['fold']
        if a.from_weights:
            nets = [load_model(f'{fam}_fold{k}', a.weights, device) for fam in ('cnx', 'r34')]
        else:
            from filament_seg.models import ResNet34UNet
            nets = []
            for cls, fam in ((ConvNeXtUNet, 'cnx'), (ResNet34UNet, 'r34')):
                net = cls(pretrained=False)
                net.load_state_dict(torch.load(a.runs/f'{fam}_fold{k}'/'best.pt', map_location='cpu', weights_only=False)['model'])
                nets.append(net.to(device).eval())
        held = sorted(records_by_source(a.cache, set(f['heldout_sources'])))[:2 if a.smoke else None]  # frames with a cache
        for s in held:
            path = a.out/f'{s}.npy'
            if path.exists():
                continue
            x = torch.from_numpy(np.load(a.cache/'images'/f'{s}.npy').astype(np.float32))[None, None].to(device)
            maps = [predict_tta(net, x)[1][0].numpy().astype(np.float16) for net in nets]
            np.save(path, np.stack([np.round(m.astype(np.float32)*255) for m in maps]).astype(np.uint8))
        print('PREP fold', k, len(held), flush=True)


class ContextDataset(Dataset):
    """One item per frame: (3, S, S) input = image + uint8 maps; target = annotator mean (occupancy, boundary)."""

    def __init__(self, cache: Path, maps: Path, groups: dict, augment: bool):
        self.cache, self.maps, self.groups, self.augment = Path(cache), Path(maps), groups, augment
        self.keys = sorted(groups)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        import cv2
        sid = self.keys[i]
        g = np.load(self.cache/'images'/f'{sid}.npy').astype(np.float32)
        m = (np.load(self.maps/f'{sid}.npy').astype(np.float32)/255-.5)/.25
        masks, edges = [], []
        for r in self.groups[sid]:
            with np.load(self.cache/r['cache_target']) as t:
                masks.append(t['mask'].astype(np.float32))
                edges.append(t['boundary'].astype(np.float32))
        y = np.stack([np.mean(masks, 0), np.mean(edges, 0)])
        x = np.concatenate([g[None], m])
        if self.augment:
            k = random.randrange(4)
            x, y = np.rot90(x, k, axes=(-2, -1)), np.rot90(y, k, axes=(-2, -1))
            if random.random() < .5:
                x, y = x[:, :, ::-1], y[:, :, ::-1]
            x, y = np.ascontiguousarray(x), np.ascontiguousarray(y)
            if random.random() < .7:  # shift-scale-rotate of all channels
                h, w = x.shape[-2:]
                M = cv2.getRotationMatrix2D((w/2-.5, h/2-.5), random.uniform(-30, 30), random.uniform(.8, 1.2))
                M[:, 2] += np.random.uniform(-.0625, .0625, 2)*w
                warp = lambda c: cv2.warpAffine(c, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
                x, y = np.stack([warp(c) for c in x]), np.stack([warp(c) for c in y])
            x[0] = photometric(x[0])  # the image channel only; the maps stay as the networks produced them
        return torch.from_numpy(np.ascontiguousarray(x)).float(), torch.from_numpy(np.ascontiguousarray(y)).float()


def train(a):
    fold = load_fold(a.folds, a.fold)
    if a.smoke:
        a.epochs, a.batch, a.workers = 1, 2, 0
    tg = records_by_source(a.cache, set(fold['train_sources']))
    ig = records_by_source(a.cache, set(fold['inner_validation_sources']))
    have = {p.stem for p in Path(a.maps).glob('*.npy')}
    tg, ig = {k: v for k, v in tg.items() if k in have}, {k: v for k, v in ig.items() if k in have}
    if a.smoke:
        tg, ig = dict(sorted(tg.items())[:4]), dict(sorted(ig.items())[:2])
    assert not (set(tg) | set(ig)) & set(fold['heldout_sources'])
    seed_everything(71+a.fold)
    device, autocast = device_and_autocast()
    torch.backends.cudnn.benchmark = True
    a.out.mkdir(parents=True, exist_ok=True)
    config = {'architecture': 'ConvNeXtUNet', 'in_chans': 3, 'fold': a.fold, 'seed': 71+a.fold, 'epochs': a.epochs, 'batch': a.batch,
              'lr': a.lr, 'ema': a.ema, 'train_sources': len(tg), 'inner_sources': len(ig), 'side': 1024,
              'folds_sha256': __import__('hashlib').sha256(a.folds.read_bytes()).hexdigest()}
    model = ConvNeXtUNet(pretrained=True, in_chans=3).to(device)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    loader = DataLoader(ContextDataset(a.cache, a.maps, tg, True), batch_size=a.batch, shuffle=True, num_workers=a.workers,
                        drop_last=True, pin_memory=device.type == 'cuda', persistent_workers=a.workers > 0)
    inner = DataLoader(ContextDataset(a.cache, a.maps, ig, False), batch_size=8, num_workers=min(a.workers, 3))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=.01)
    total = a.epochs*len(loader)
    warm = max(1, min(200, total//10))
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s+1)/warm if s < warm else .5*(1+math.cos(math.pi*(s-warm)/max(1, total-warm))))
    best, steps, hist, t0 = float('inf'), 0, [], time.monotonic()
    for epoch in range(a.epochs):
        model.train()
        losses = []
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast():
                z = model(x)
            loss = segmentation_loss(z, y, 1.)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3., error_if_nonfinite=True)
            opt.step()
            sch.step()
            steps += 1
            losses.append(float(loss.detach()))
            d = min(a.ema, (1+steps)/(10+steps))
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.lerp_(pm, 1-d)
                for be, bm in zip(ema.buffers(), model.buffers()):
                    be.copy_(bm)
        total_loss, n = 0., 0
        with torch.no_grad():
            for x, y in inner:
                x, y = x.to(device), y.to(device)
                with autocast():
                    z = ema(x)
                total_loss += float(segmentation_loss(z.float(), y, 1.))*len(x)
                n += len(x)
        hist.append({'epoch': epoch, 'loss': total_loss/max(1, n), 'train_loss': float(np.mean(losses)), 'seconds': round(time.monotonic()-t0, 1)})
        print(json.dumps(hist[-1]), flush=True)
        if hist[-1]['loss'] < best:
            best = hist[-1]['loss']
            atomic_save({'model': ema.state_dict(), 'config': config, 'epoch': epoch}, a.out/'best.pt')
    (a.out/'history.json').write_text(json.dumps({'config': config, 'history': hist}, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['prep', 'train'])
    ap.add_argument('--cache', type=Path, required=True, help='prepare_data.py output for side 1024')
    ap.add_argument('--folds', type=Path, default=Path('configs/folds_full.json'))
    ap.add_argument('--runs', type=Path, help='prep: folder with cnx_fold{k}/ and r34_fold{k}/ best.pt')
    ap.add_argument('--from-weights', action='store_true', help='prep: use the released weights/ instead of --runs')
    ap.add_argument('--weights', type=Path, default=Path('weights'))
    ap.add_argument('--maps', type=Path, help='train: prep output')
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--ema', type=float, default=.999)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--smoke', action='store_true', help='prep: 2 frames per fold; train: 1 epoch on 4 frames')
    a = ap.parse_args()
    {'prep': prep, 'train': train}[a.cmd](a)


if __name__ == '__main__':
    main()
