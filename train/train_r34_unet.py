"""Train one fold of the ResNet34 U-Net (the "voter" weights in weights/r34_fold*.safetensors).

Two stages, both selecting the checkpoint with the lowest loss on the fold's
inner month slice; one item per annotator record, sampled with weight
1/(records of its frame) so every frame counts equally:
  A: 768 input, ImageNet init, 45 epochs, batch 8, AdamW lr 3e-4, cosine
  B: 1024 input, init from A's best, 12 epochs, batch 4, lr 1e-4, BatchNorm
     statistics frozen (affine parameters still train)
Loss .5 BCE (pos_weight 3) + .5 soft Dice + .15 boundary BCE, grad-clip 3,
bf16 autocast. ~1 h per fold on an H200.

    python train/train_r34_unet.py --cache768 work/prepared/768 --cache1024 work/prepared/1024 --fold 0 --out work/runs/r34_fold0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (RecordDataset, atomic_save, device_and_autocast, load_fold, records_by_source,  # noqa: E402
                    seed_everything, segmentation_loss)
from filament_seg.models import ResNet34UNet  # noqa: E402


def freeze_bn_statistics(model):
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def run_stage(name, model, cache, train_ids, inner_ids, out, *, epochs, batch, lr, frozen_bn, workers, smoke):
    device, autocast = device_and_autocast()
    rows = [r for g in records_by_source(cache, set(train_ids)).values() for r in g]
    inner = [r for g in records_by_source(cache, set(inner_ids)).values() for r in g]
    if smoke:
        rows, inner = rows[:4], inner[:2]
    per_source = {}
    for r in rows:
        per_source[r['source_id']] = per_source.get(r['source_id'], 0)+1
    sampler = WeightedRandomSampler([1/per_source[r['source_id']] for r in rows], len(rows), replacement=True)
    loader = DataLoader(RecordDataset(cache, rows, True), batch_size=batch, sampler=sampler, num_workers=workers,
                        pin_memory=device.type == 'cuda')
    inner_loader = DataLoader(RecordDataset(cache, inner, False), batch_size=8, num_workers=workers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.01)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best, history, start = float('inf'), [], time.monotonic()
    for epoch in range(epochs):
        model.train()
        if frozen_bn:
            freeze_bn_statistics(model)
        losses = []
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                z = model(x)
            loss = segmentation_loss(z.float(), y, 3.)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3., error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        total, n = 0., 0
        with torch.no_grad():
            for x, y in inner_loader:
                x, y = x.to(device), y.to(device)
                with autocast():
                    z = model(x)
                total += float(segmentation_loss(z.float(), y, 3.))*len(x)
                n += len(x)
        history.append({'stage': name, 'epoch': epoch, 'train_loss': float(np.mean(losses)), 'inner_loss': total/max(n, 1),
                        'seconds': round(time.monotonic()-start, 1)})
        print(json.dumps(history[-1]), flush=True)
        if history[-1]['inner_loss'] < best:
            best = history[-1]['inner_loss']
            atomic_save({'model': model.state_dict(), 'epoch': epoch}, out/f'{name}_best.pt')
        schedule.step()
    return history


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cache768', type=Path, required=True)
    ap.add_argument('--cache1024', type=Path, required=True)
    ap.add_argument('--folds', type=Path, default=Path('configs/folds_full.json'))
    ap.add_argument('--fold', type=int, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--epochs-a', type=int, default=45)
    ap.add_argument('--epochs-b', type=int, default=12)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--smoke', action='store_true', help='1 epoch per stage on 4 records')
    a = ap.parse_args()
    if a.smoke:
        a.epochs_a = a.epochs_b = 1
        a.workers = 0
    fold = load_fold(a.folds, a.fold)
    a.out.mkdir(parents=True, exist_ok=True)
    seed_everything(11+a.fold)
    torch.backends.cudnn.benchmark = True
    device, _ = device_and_autocast()
    model = ResNet34UNet(pretrained=True).to(device)
    common = dict(train_ids=fold['train_sources'], inner_ids=fold['inner_validation_sources'], out=a.out,
                  workers=a.workers, smoke=a.smoke)
    hist = run_stage('stageA', model, a.cache768, epochs=a.epochs_a, batch=8, lr=3e-4, frozen_bn=False, **common)
    model.load_state_dict(torch.load(a.out/'stageA_best.pt', map_location='cpu')['model'])
    hist += run_stage('stageB', model, a.cache1024, epochs=a.epochs_b, batch=4, lr=1e-4, frozen_bn=True, **common)
    final = torch.load(a.out/'stageB_best.pt', map_location='cpu')
    config = {'architecture': 'ResNet34UNet', 'fold': a.fold, 'seed': 11+a.fold, 'side': 1024, 'epochs_a': a.epochs_a,
              'epochs_b': a.epochs_b, 'train_sources': len(fold['train_sources']),
              'inner_sources': len(fold['inner_validation_sources']), 'precision': 'bf16 autocast'}
    atomic_save({'model': final['model'], 'config': config, 'epoch': final['epoch']}, a.out/'best.pt')
    (a.out/'history.json').write_text(json.dumps({'config': config, 'history': hist}, indent=1))


if __name__ == '__main__':
    main()
