"""Train the 5 fold crop CNNs of the verifier and their out-of-fold scores.

For fold k the CNN trains on the candidates of all frames not held out in
fold k and scores the held-out ones, so every candidate gets an out-of-fold
score (the GBM's 47th input). Recipe: timm resnet18.a1_in1k with a 5-channel
stem + log crop side, soft BCE on u plus .5 x soft BCE on the match fraction,
12 epochs, batch 128, AdamW lr 1e-3 wd .05, one-cycle, dihedral augmentation,
bf16 autocast, 4-rotation TTA at scoring. ~1 min per fold on an H200.

    python train/train_crop_cnn.py --data work/verifier_data.npz --out work/runs/crop_cnn
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import device_and_autocast  # noqa: E402
from filament_seg.models import CropVerifierCNN  # noqa: E402


@torch.no_grad()
def score(model, crops, sides, device, autocast):
    model.eval()
    out = []
    for i in range(0, len(crops), 512):
        x = crops[i:i+512].to(device).float()/255
        s = sides[i:i+512].to(device)
        acc = 0
        with autocast():
            for k in range(4):
                acc = acc+torch.sigmoid(model(torch.rot90(x, k, (-2, -1)), s).float())
        out.append((acc/4).cpu())
    return torch.cat(out).numpy()[:, 0] if out else np.zeros(0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', type=Path, required=True, help='build_verifier_data.py output')
    ap.add_argument('--folds', type=Path, default=Path('configs/folds_full.json'))
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--smoke', action='store_true', help='1 epoch, batch 8')
    a = ap.parse_args()
    if a.smoke:
        a.epochs, a.batch = 1, 8
    d = np.load(a.data)
    owner = np.repeat(d['keys'], d['counts'])
    X, S = torch.from_numpy(d['crops']), torch.from_numpy(d['sides'])
    Y = torch.from_numpy(d['targets']).float()  # [match, u]
    device, autocast = device_and_autocast()
    torch.backends.cudnn.benchmark = True
    a.out.mkdir(parents=True, exist_ok=True)
    oof = np.full(len(X), np.nan)
    for fold in json.loads(a.folds.read_text())['folds']:
        k = fold['fold']
        held = np.isin(owner, fold['heldout_sources'])
        idx = np.flatnonzero(~held)
        torch.manual_seed(100+k)
        np.random.seed(100+k)
        model = CropVerifierCNN(pretrained=True).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=.05)
        steps = max(1, a.epochs*(len(idx)//a.batch))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=.1)
        for epoch in range(a.epochs):
            model.train()
            perm = idx[np.random.permutation(len(idx))]
            total = 0.
            for i in range(0, len(perm)-a.batch+1, a.batch):
                b = perm[i:i+a.batch]
                x = torch.rot90(X[b].to(device).float()/255, np.random.randint(4), (-2, -1))
                if np.random.rand() < .5:
                    x = torch.flip(x, (-1,))
                with autocast():
                    z = model(x, S[b].to(device)).float()
                y = Y[b].to(device)
                loss = F.binary_cross_entropy_with_logits(z[:, 0], y[:, 1])+.5*F.binary_cross_entropy_with_logits(z[:, 1], y[:, 0])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                total += float(loss.detach())
            print(f'fold {k} epoch {epoch} loss {total/max(1, len(perm)//a.batch):.4f}', flush=True)
        oof[held] = score(model, X[held], S[held], device, autocast)
        torch.save(model.state_dict(), a.out/f'cnn_fold{k}.pt')
    np.savez(a.out/'cnn_oof_scores.npz', keys=d['keys'], counts=d['counts'], oof_u=oof)


if __name__ == '__main__':
    main()
