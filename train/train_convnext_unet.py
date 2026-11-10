"""Train one fold of the ConvNeXt-T U-Net segmenter (the weights in weights/cnx_fold*.safetensors).

Recipe: ImageNet-22k ConvNeXt-T encoder, 1024 input, soft target = mean of the
frame's annotator occupancy maps, loss = .5 BCE (pos_weight 1) + .5 soft Dice
+ .15 boundary BCE, AdamW lr 2e-4 wd .01, 200-step warm-up + cosine,
batch 8, 120 epochs, grad-clip 3, bf16 autocast, EMA .999 (warm-started).
Augmentation: dihedral, shift-scale-rotate (p .7), gamma and gain jitter.
The EMA weights with the lowest loss on the fold's inner month slice are saved.
Training used one H200 GPU, ~1.3-1.5 h per fold (5 folds ran concurrently).

    python train/train_convnext_unet.py --cache work/prepared/1024 --fold 0 --out work/runs/cnx_fold0
    python train/train_convnext_unet.py ... --smoke     # 1 epoch on 4 frames (CPU OK)
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (SourceMeanDataset, atomic_save, device_and_autocast, load_fold, records_by_source,  # noqa: E402
                    seed_everything, segmentation_loss)
from filament_seg.models import ConvNeXtUNet  # noqa: E402


@torch.no_grad()
def inner_loss(model, loader, device, autocast, pos_weight):
    model.eval()
    total, n = 0., 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with autocast():
            z = model(x)
        total += float(segmentation_loss(z.float(), y, pos_weight))*len(x)
        n += len(x)
    return total/max(n, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cache', type=Path, required=True, help='prepare_data.py output for side 1024')
    ap.add_argument('--folds', type=Path, default=Path('configs/folds_full.json'))
    ap.add_argument('--fold', type=int, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--epochs', type=int, default=120)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--pos-weight', type=float, default=1.)
    ap.add_argument('--ema', type=float, default=.999)
    ap.add_argument('--seed', type=int, default=31)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--encoder', default='convnext_tiny.fb_in22k_ft_in1k')
    ap.add_argument('--smoke', action='store_true', help='1 epoch, 4 train + 2 inner frames, batch 2')
    a = ap.parse_args()
    fold = load_fold(a.folds, a.fold)
    train_ids, inner_ids = fold['train_sources'], fold['inner_validation_sources']
    if a.smoke:
        a.epochs, a.batch, a.workers = 1, 2, 0
    train_groups = records_by_source(a.cache, set(train_ids))
    inner_groups = records_by_source(a.cache, set(inner_ids))
    if a.smoke:
        train_groups = dict(sorted(train_groups.items())[:4])
        inner_groups = dict(sorted(inner_groups.items())[:2])
    seed_everything(a.seed+a.fold)
    device, autocast = device_and_autocast()
    torch.backends.cudnn.benchmark = True
    a.out.mkdir(parents=True, exist_ok=True)
    config = {'architecture': 'ConvNeXtUNet', 'encoder': a.encoder, 'fold': a.fold, 'seed': a.seed+a.fold, 'side': 1024,
              'epochs': a.epochs, 'batch': a.batch, 'lr': a.lr, 'pos_weight': a.pos_weight, 'ema': a.ema,
              'train_sources': len(train_groups), 'inner_sources': len(inner_groups), 'precision': 'bf16 autocast'}
    model = ConvNeXtUNet(a.encoder, pretrained=True).to(device)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    loader = DataLoader(SourceMeanDataset(a.cache, train_groups, True), batch_size=a.batch, shuffle=True,
                        num_workers=a.workers, drop_last=True, pin_memory=device.type == 'cuda')
    inner = DataLoader(SourceMeanDataset(a.cache, inner_groups, False), batch_size=8, num_workers=min(a.workers, 4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=.01)
    total_steps = a.epochs*len(loader)
    warm = max(1, min(200, total_steps//10))
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: (s+1)/warm if s < warm else
                                                 .5*(1+math.cos(math.pi*(s-warm)/max(1, total_steps-warm))))
    best, steps, history, start = float('inf'), 0, [], time.monotonic()
    for epoch in range(a.epochs):
        model.train()
        losses = []
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                z = model(x)
            loss = segmentation_loss(z, y, a.pos_weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3., error_if_nonfinite=True)
            optimizer.step()
            schedule.step()
            steps += 1
            losses.append(float(loss.detach()))
            decay = min(a.ema, (1+steps)/(10+steps))
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.lerp_(pm, 1-decay)
                for be, bm in zip(ema.buffers(), model.buffers()):
                    be.copy_(bm)
        val = inner_loss(ema, inner, device, autocast, a.pos_weight)
        history.append({'epoch': epoch, 'train_loss': float(np.mean(losses)), 'inner_loss': val,
                        'seconds': round(time.monotonic()-start, 1)})
        print(json.dumps(history[-1]), flush=True)
        if val < best:
            best = val
            atomic_save({'model': ema.state_dict(), 'config': config, 'epoch': epoch}, a.out/'best.pt')
    (a.out/'history.json').write_text(json.dumps({'config': config, 'history': history}, indent=1))


if __name__ == '__main__':
    main()
