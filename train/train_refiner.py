"""Instance refiners of the final pipeline (weights/refiner_fold*.safetensors): build the crop bank, train one fold.

Recipe "ab30" (the released weights). For every labelled frame of fold k, everything comes from fold-k networks that
never saw the frame: the fused D4 map (ConvNeXt k + specialist k), its candidates and nz1 outlines ("seeds"), and the
FZ verifier score u_FZ cross-fitted over the folds (5 bags per fold, the verifier recipe of train_gbm.py).

Rows of the bank (refiner.py builds the crops; 4 uint8 channels: image, fused map, seed, other candidates with
u_FZ >= .10 weighted by u_FZ):
  fz      every nz1 seed with at least one accepted annotator record
  trunc   six synthetic truncations of each such seed: a random 40-80% of the seed along its long axis (one end, the
          other end or a middle part), sometimes eroded by 1-3 px, sometimes reduced to its largest piece; the crop is
          re-centred on the cut seed, the target stays the consensus of the full seed, and the cut-away candidate is
          hidden from the neighbour channel
  extra   natural seeds of other decodings of the same frame: components of the fused map at .5 / .6 / .7 / .3 and of
          the single-family maps (ConvNeXt only, specialist only) at .4 / .5 / .6 / .3; a seed is dropped when no
          record accepts it or when it repeats (IoU > .9) an earlier seed
Target: for each annotator record, the annotated instance with the highest IoU with the seed, accepted at IoU >= .20;
the target is the mean over the accepted records (records that did not draw the filament are left out).

Training (fold k of configs/folds_full.json; selection on the inner slice's fz rows): each epoch uses every fz row
of the training frames, each replaced with probability .3 by one of its truncations, plus as many extra rows as there
are fz rows, drawn by cycling a random permutation. RefineUNet (ImageNet ConvNeXt-T, 4-channel stem), loss .5 BCE + .5
soft Dice, row weight = number of annotator records, 40 epochs, batches of 48 (128 x 384) / 16 (192 x 768), AdamW
lr 3e-4 (encoder x .5) wd .01, 200-step warm-up + cosine, grad-clip 3, bf16 autocast, EMA .998 (warm-started);
augmentation on the GPU: flips, rotation +-5 deg, scale +-8 %, shifts inside the stored margin (30 % of crops only
integer shifts), gain / bias / gamma jitter of the image channel. The EMA weights with the lowest inner loss are saved.
About 10-18 minutes per fold on one RTX 5090.

    python train/train_refiner.py bank --data work/verifier_data_fz.npz --cnn work/runs/crop_cnn_fz/cnn_oof_scores.npz --out work/refiner_bank
    python train/train_refiner.py train --bank work/refiner_bank --fold 0 --out work/runs/refiner_fold0
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import pickle
import random
import sys
import time
import zlib

import numpy as np
from scipy import ndimage as ndi
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from filament_seg import refiner as R  # noqa: E402

TBAR, RMAX, KTRUNC, DUP_IOU, SAME_OBJ, U_MIN = .20, 3, 6, .9, .5, .10
EIGHT = np.ones((3, 3), bool)
DISK = {r: (lambda yy, xx: yy*yy+xx*xx <= r*r+.5)(*np.ogrid[-r:r+1, -r:r+1]) for r in range(4)}
KINDS = [('t50', 'fused', .5), ('t60', 'fused', .6), ('t70', 'fused', .7), ('t30', 'fused', .3), ('base', 'base', .4), ('spec', 'spec', .4),
         ('base50', 'base', .5), ('spec50', 'spec', .5), ('base60', 'base', .6), ('spec60', 'spec', .6), ('base30', 'base', .3), ('spec30', 'spec', .3)]


# ------------------------------------------------------------------ bank
def crossfit_u(data, cnn, folds):
    """Cross-fitted FZ verifier score per frame: for fold k, 5 bags fitted on the other folds' frames (seeds 7 + k)."""
    from train_gbm import fit_bag, frame_rows
    rows = frame_rows(data['keys'], data['counts'], data['features'], data['targets'], cnn['oof_u'])
    out = {}
    for f in folds:
        held = set(f['heldout_sources'])
        models = fit_bag([r for r in rows if r[0] not in held], 5, 7+f['fold'], 3)
        for k, x, _ in rows:
            if k in held:
                out[k] = np.clip(np.mean([m.predict(x) for m in models], 0), 0, 1) if len(x) else np.zeros(0)
    return out


def gt_masks(rles):
    from pycocotools import mask as cm
    out = []
    for r in rles:
        m = cm.decode(r).astype(bool)
        if not m.any():
            out.append(None)
            continue
        sl = ndi.find_objects(m.astype(np.uint8))[0]
        out.append((sl[0].start, sl[1].start, m[sl].copy(), int(m.sum())))
    return out


def match(lab, n, areas, recs):
    out = []
    for gm in recs:
        iou, inter, ga = np.zeros((len(gm), n)), np.zeros((len(gm), n)), np.zeros(len(gm))
        for gi, g in enumerate(gm):
            if g is None:
                continue
            y0, x0, m, a = g
            ga[gi] = a
            sub = lab[y0:y0+m.shape[0], x0:x0+m.shape[1]][m]
            ids, cnt = np.unique(sub[sub > 0], return_counts=True)
            inter[gi, ids-1], iou[gi, ids-1] = cnt, cnt/(a+areas[ids-1]-cnt)
        out.append((iou, inter, ga))
    return out


def accepted_gt(mt, j):
    """Per record (<= 3): the GT index with the highest IoU with seed j if that IoU >= TBAR, else -1."""
    sel = -np.ones(RMAX, np.int64)
    for r, (iou, _, _) in enumerate(mt):
        if iou.shape[0] and iou[:, j].max() >= TBAR:
            sel[r] = int(np.argmax(iou[:, j]))
    return sel


def truncate(mask, y0, x0, rng):
    (cx, cy), (dx, dy), _, _ = R.seed_rect(mask, y0, x0)
    ys, xs = np.nonzero(mask)
    s = (xs+x0-cx)*dx+(ys+y0-cy)*dy
    lo, hi = float(s.min()), float(s.max())
    span, area0 = hi-lo, int(mask.sum())
    for _ in range(4):
        f, mode = float(rng.uniform(.4, .8)), rng.choice(3, p=[.4, .4, .2])
        if mode == 0:
            a, b = lo-1, lo+f*span
        elif mode == 1:
            a, b = hi-f*span, hi+1
        else:
            a = lo+float(rng.uniform(0, 1-f))*span
            b = a+f*span
        keep = (s >= a) & (s <= b)
        m = np.zeros_like(mask)
        m[ys[keep], xs[keep]] = True
        if rng.random() < .5:
            er = int(rng.integers(1, 4))
            while er > 0:
                e = ndi.binary_erosion(m, DISK[er])
                if int(e.sum()) >= max(40, .35*int(m.sum())):
                    m = e
                    break
                er -= 1
        lab, k = ndi.label(m, EIGHT)
        if k > 1 and rng.random() < .5:
            m = lab == int(np.argmax(np.bincount(lab.ravel())[1:]))+1
        if 40 <= int(m.sum()) <= .92*area0:
            return m
    return None


def crop(ctx, seed_fn, mask_box, hide):
    """Crop (bucket, M, x uint8 [4, H, W]) of a seed given as a function of a native box -> bool mask."""
    center, d, length, width = R.seed_rect(*mask_box)
    bucket, scale = R.choose_bucket(length, width)
    (ih, iw), (my, mx) = R.BUCKETS[bucket]['inner'], R.BUCKETS[bucket]['margin']
    shw = (ih+2*my, iw+2*mx)
    M = R.affine(center, d, scale, shw)
    (y0, y1, x0, x1), Mr = R.crop_roi(M, shw)
    seed = seed_fn(y0, y1, x0, x1)
    w = ctx['weights'].copy()
    if len(hide):
        w[np.asarray(hide, np.int64)+1] = 0
    nbr = w[ctx['seeds'][y0:y1, x0:x1]]
    nbr[seed] = 0
    x = np.stack([R.gray_q8(R.warp(ctx['gn'][y0:y1, x0:x1], Mr, shw)), R.q8(R.warp(ctx['fg'][y0:y1, x0:x1], Mr, shw)),
                  R.q8(R.warp(seed, Mr, shw)), R.q8(R.warp(nbr, Mr, shw))])
    return bucket, M, (y0, y1, x0, x1), Mr, shw, x


def target(ctx, sel, roi, Mr, shw):
    y0, y1, x0, x1 = roi
    t, k = np.zeros(shw, np.float32), 0
    for r in np.flatnonzero(sel >= 0):
        gy, gx, m, _ = ctx['recs'][r][sel[r]]
        g = np.zeros((y1-y0, x1-x0), np.float32)
        ya, yb, xa, xb = max(gy, y0), min(gy+m.shape[0], y1), max(gx, x0), min(gx+m.shape[1], x1)
        if yb > ya and xb > xa:
            g[ya-y0:yb-y0, xa-x0:xb-x0] = m[ya-gy:yb-gy, xa-gx:xb-gx]
        t += R.q8(R.warp(g, Mr, shw)).astype(np.float32)/255
        k += 1
    return R.q8(t/max(k, 1))


def bank_frame(key, fold, pred, gray, gt_records, u, settings, nz_step):
    """All bank rows of one labelled frame (fold-k networks only)."""
    from filament_seg.candidates import candidate_labels
    from filament_seg.data import normalize_image
    from filament_seg.inference import specialist_mean, to_native
    from filament_seg.postprocess import InstanceState, apply_steps
    maps = pred(pred.path_of[key], keep_folds=True)
    fg = maps['decode'][0]
    base = to_native(maps['cnx_folds'])[0]
    spec = specialist_mean(pred.last_specialist)
    labels = candidate_labels(fg, settings)
    n = int(labels.max())
    assert n == len(u), f'{key}: {n} candidates, verifier table {len(u)}'
    seeds = apply_steps(InstanceState(labels, np.ones(n, bool), fg, gray), [nz_step]).labels
    areas = np.bincount(seeds.ravel(), minlength=n+1)[1:]
    recs = [gt_masks(rles) for _, rles in gt_records][:RMAX]
    ctx = {'seeds': seeds, 'gn': normalize_image(gray), 'fg': fg, 'recs': recs,
           'weights': np.r_[0., np.where(u >= U_MIN, u, 0.)].astype(np.float32)}
    rows = []
    mt = match(seeds, n, areas, recs)
    objects = ndi.find_objects(seeds, max_label=n)
    for j in range(n):
        sel = accepted_gt(mt, j)
        if not (sel >= 0).any():
            continue
        b = objects[j]
        mk, by, bx = seeds[b] == j+1, b[0].start, b[1].start
        bucket, M, roi, Mr, shw, x = crop(ctx, lambda y0, y1, x0, x1: seeds[y0:y1, x0:x1] == j+1, (mk, by, bx), [])
        rows.append({'kind': 'fz', 'cand': j, 'bucket': bucket, 'x': x, 't': target(ctx, sel, roi, Mr, shw), 'nrec': len(recs)})
        for kk in range(KTRUNC):
            tm = truncate(mk, by, bx, np.random.default_rng(zlib.crc32(f'{key}:{j}:{kk}'.encode())))
            if tm is None:
                continue

            def cut(y0, y1, x0, x1, tm=tm, by=by, bx=bx):
                out = np.zeros((y1-y0, x1-x0), bool)
                ya, yb, xa, xb = max(by, y0), min(by+tm.shape[0], y1), max(bx, x0), min(bx+tm.shape[1], x1)
                if yb > ya and xb > xa:
                    out[ya-y0:yb-y0, xa-x0:xb-x0] = tm[ya-by:yb-by, xa-bx:xb-bx]
                return out
            bucket, M, roi, Mr, shw, x = crop(ctx, cut, (tm, by, bx), [j])
            rows.append({'kind': 'trunc', 'cand': j, 'bucket': bucket, 'x': x, 't': target(ctx, sel, roi, Mr, shw), 'nrec': len(recs)})
    existing = [(seeds, areas)]
    sources = {'fused': fg, 'base': base, 'spec': spec}
    from filament_seg.candidates import CandidateSettings
    for kind, name, thr in KINDS:
        lab = candidate_labels(sources[name], CandidateSettings(thr, settings.min_area, True))
        nk = int(lab.max())
        ar = np.bincount(lab.ravel(), minlength=nk+1)[1:]
        mk_ = match(lab, nk, ar, recs)
        ob = ndi.find_objects(lab, max_label=nk)
        keep = np.zeros(nk, bool)
        for j in range(nk):
            sel = accepted_gt(mk_, j)
            if not (sel >= 0).any():
                continue
            b = ob[j]
            mk = lab[b] == j+1
            dup, same = False, []
            for li, (el, ea) in enumerate(existing):
                ids, cnt = np.unique(el[b][mk], return_counts=True)
                ok = ids > 0
                ids, cnt = ids[ok], cnt[ok]
                if len(ids) and float((cnt/(ar[j]+ea[ids-1]-cnt)).max()) > DUP_IOU:
                    dup = True
                    break
                if li == 0:  # candidates that are the same object stay out of the neighbour channel
                    same = [int(i)-1 for i, c in zip(ids, cnt) if c >= SAME_OBJ*min(ar[j], ea[i-1])]
            if dup:
                continue
            keep[j] = True
            bucket, M, roi, Mr, shw, x = crop(ctx, lambda y0, y1, x0, x1, lab=lab, j=j: lab[y0:y1, x0:x1] == j+1,
                                              (mk, b[0].start, b[1].start), same)
            rows.append({'kind': kind, 'cand': -1, 'bucket': bucket, 'x': x, 't': target(ctx, sel, roi, Mr, shw), 'nrec': len(recs)})
        mp = np.zeros(nk+1, np.int32)
        mp[1:][keep] = np.arange(1, int(keep.sum())+1)
        existing.append((mp[lab], ar[keep]))
    for r in rows:
        r['key'], r['fold'] = key, fold
    return rows


def cmd_bank(a):
    from filament_seg.candidates import CandidateSettings
    from filament_seg.config import data_path, load_config, resolve
    from filament_seg.data import list_images, load_ground_truth, read_gray
    from filament_seg.inference import EnsemblePredictor
    cfg = load_config(a.config)
    wd = resolve(cfg['weights']['dir'])
    folds = json.loads(resolve(cfg['data']['folds']).read_text())['folds']
    data, cnn = np.load(a.data), np.load(a.cnn)
    u_oof = crossfit_u(data, cnn, folds)
    paths = list_images(data_path(cfg, 'train_images'))
    settings, nz_step = CandidateSettings.from_config(cfg), cfg['postprocess']['steps'][0]
    a.out.mkdir(parents=True, exist_ok=True)
    meta, store = [], {bk: {'x': [], 't': []} for bk in 'AB'}
    for f in folds:
        k = f['fold']
        pred = EnsemblePredictor.from_weights(cfg, wd, [f'cnx_fold{k}'], [f'r34_fold{k}'], [f'spec_fold{k}'])
        pred.path_of = paths
        orig = pred.specialist_fold_maps

        def keep_last(gray, orig=orig, pred=pred):  # remember the specialist D4 map for the single-family seeds
            s, d = orig(gray)
            pred.last_specialist = d
            return s, d
        pred.specialist_fold_maps = keep_last
        keys = sorted(f['heldout_sources'])[:2 if a.smoke else None]
        gt = load_ground_truth(data_path(cfg, 'train_annotations'), set(keys))
        for key in keys:
            for r in bank_frame(key, k, pred, read_gray(paths[key]), gt[key], u_oof[key], settings, nz_step):
                store[r['bucket']]['x'].append(r.pop('x'))
                store[r['bucket']]['t'].append(r.pop('t'))
                r['row'] = len(store[r['bucket']]['x'])-1
                meta.append(r)
        print('BANK fold', k, len(meta), 'rows so far', flush=True)
    for bk in 'AB':
        (ih, iw), (my, mx) = R.BUCKETS[bk]['inner'], R.BUCKETS[bk]['margin']
        np.save(a.out/f'{bk}_x.npy', np.asarray(store[bk]['x'], np.uint8).reshape(-1, 4, ih+2*my, iw+2*mx))
        np.save(a.out/f'{bk}_t.npy', np.asarray(store[bk]['t'], np.uint8).reshape(-1, 1, ih+2*my, iw+2*mx))
    pickle.dump(meta, open(a.out/'meta.pkl', 'wb'))
    print('BANK_DONE', len(meta), {kind: sum(r['kind'] == kind for r in meta) for kind in ['fz', 'trunc']+[k for k, _, _ in KINDS]}, flush=True)


# ------------------------------------------------------------------ training
def augment(inp, inner, gen, p_plain=.3):
    """inp [B, C, Hs, Ws] -> [B, C, Hi, Wi]: flips + (rotation, scale, shift) jitter inside the stored margin."""
    B, _, Hs, Ws = inp.shape
    Hi, Wi = inner
    dev = inp.device
    r = lambda *s: torch.rand(*s, device=dev, generator=gen)
    plain = r(B) < p_plain
    rot = torch.where(plain, torch.zeros(B, device=dev), (r(B)*2-1)*math.radians(5.))
    sc = torch.where(plain, torch.ones(B, device=dev), torch.exp((r(B)*2-1)*math.log(1.08)))
    tx, ty = (r(B)*2-1)*20, (r(B)*2-1)*8
    tx, ty = torch.where(plain, tx.round(), tx), torch.where(plain, ty.round(), ty)
    fx, fy = torch.where(r(B) < .5, -1., 1.), torch.where(r(B) < .5, -1., 1.)
    cos, sin = torch.cos(rot)/sc, torch.sin(rot)/sc
    a, b, c, d = cos*fx, -sin*fy, sin*fx, cos*fy
    theta = torch.stack([torch.stack([a*Wi/Ws, b*Hi/Ws, tx/(Ws/2)], 1), torch.stack([c*Wi/Hs, d*Hi/Hs, ty/(Hs/2)], 1)], 1)
    grid = F.affine_grid(theta, (B, inp.shape[1], Hi, Wi), align_corners=False)
    return F.grid_sample(inp, grid, mode='bilinear', padding_mode='zeros', align_corners=False)


def photometric(g, gen):
    """Gain / bias / gamma jitter of the image channel (stored as uint8 over [-2, 2])."""
    B, dev = g.shape[0], g.device
    r = lambda: torch.rand(B, 1, 1, 1, device=dev, generator=gen)
    v = g*4-2
    v = v*(.9+.2*r())+(r()*.2-.1)
    gam = torch.where(r() < .5, torch.exp((r()*2-1)*.3), torch.ones(B, 1, 1, 1, device=dev))
    u = torch.clamp(v*.25+.5, 0, 1)**gam
    v = torch.where((v > -2) & (v < 2), (u-.5)/.25, v)
    return (torch.clamp(v, -2, 2)+2)/4


def loss_fn(logit, t, w):
    logit = logit.float()
    bce = F.binary_cross_entropy_with_logits(logit, t, reduction='none').mean((1, 2, 3))
    p = torch.sigmoid(logit)
    dice = 1-(2*(p*t).sum((1, 2, 3))+1)/((p+t).sum((1, 2, 3))+1)
    return ((.5*bce+.5*dice)*w).sum()/w.sum()


def cmd_train(a):
    from filament_seg.models import RefineUNet
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = 9000+a.fold
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    rs = np.random.default_rng(seed)
    fold = json.loads(a.folds.read_text())['folds'][a.fold]
    tr_keys, in_keys = set(fold['train_sources']), set(fold['inner_validation_sources'])
    assert not (tr_keys | in_keys) & (set(fold['heldout_sources']) | set(fold.get('purged_sources', []))), 'cross-fitting violated'
    meta = pickle.load(open(a.bank/'meta.pkl', 'rb'))
    X = {bk: np.load(a.bank/f'{bk}_x.npy', mmap_mode='r') for bk in 'AB'}
    T = {bk: np.load(a.bank/f'{bk}_t.npy', mmap_mode='r') for bk in 'AB'}
    fz = [r for r in meta if r['kind'] == 'fz' and r['key'] in tr_keys]
    inner = [r for r in meta if r['kind'] == 'fz' and r['key'] in in_keys]
    truncs = {}
    for r in meta:
        if r['kind'] == 'trunc' and r['key'] in tr_keys:
            truncs.setdefault((r['key'], r['cand']), []).append(r)
    extra = [r for r in meta if r['kind'] not in ('fz', 'trunc') and r['key'] in tr_keys]
    epochs, cycle = [], []
    for _ in range(1 if a.smoke else a.epochs):
        sel = []
        for r in fz:
            tl = truncs.get((r['key'], r['cand']))
            sel.append(tl[int(rs.integers(len(tl)))] if tl and rs.random() < a.trunc_p else r)
        for _ in range(int(round(a.alt_ratio*len(fz))) if extra else 0):
            if not cycle:
                cycle = [extra[i] for i in rs.permutation(len(extra))]
            sel.append(cycle.pop())
        epochs.append(sel)
    bs = {'A': a.bs_a, 'B': a.bs_b}
    total = sum(math.ceil(sum(r['bucket'] == bk for r in sel)/bs[bk]) for sel in epochs for bk in 'AB')
    model = RefineUNet(pretrained=True, in_chans=4).to(dev).to(memory_format=torch.channels_last)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    enc = [p for n, p in model.named_parameters() if n.startswith('encoder.')]
    rest = [p for n, p in model.named_parameters() if not n.startswith('encoder.')]
    opt = torch.optim.AdamW([{'params': enc, 'lr': a.lr*.5}, {'params': rest, 'lr': a.lr}], weight_decay=.01)
    warm = min(200, max(1, total//10))
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s+1)/warm if s < warm else .5*(1+math.cos(math.pi*(s-warm)/max(1, total-warm))))
    amp = dev.type == 'cuda'
    prep = lambda x: torch.cat([x[:, :1]*4-2, x[:, 1:]], 1)

    def load(rows, bk):
        idx = np.asarray([r['row'] for r in rows], np.int64)
        o = np.argsort(idx, kind='stable')
        x, t = np.empty((len(idx),)+X[bk].shape[1:], np.uint8), np.empty((len(idx),)+T[bk].shape[1:], np.uint8)
        x[o], t[o] = X[bk][idx[o]], T[bk][idx[o]]
        w = np.asarray([max(1, r['nrec']) for r in rows], np.float32)
        return torch.from_numpy(x).to(dev), torch.from_numpy(t).to(dev), torch.from_numpy(w).to(dev)

    @torch.no_grad()
    def validate(net):
        tot, n = 0., 0.
        for bk in 'AB':
            rows = [r for r in inner if r['bucket'] == bk]
            (h, w), (my, mx) = R.BUCKETS[bk]['inner'], R.BUCKETS[bk]['margin']
            for i in range(0, len(rows), 64):
                x, t, wt = load(rows[i:i+64], bk)
                x, t = (x.float()/255)[..., my:my+h, mx:mx+w], (t.float()/255)[..., my:my+h, mx:mx+w]
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
                    z = net(prep(x))
                tot += float(loss_fn(z, t, wt))*float(wt.sum())
                n += float(wt.sum())
        return tot/max(n, 1e-9)

    config = {'world': 'full', 'fold': a.fold, 'in_ch': 4, 'tbar': TBAR, 'epochs': a.epochs, 'trunc_p': a.trunc_p, 'alt_ratio': a.alt_ratio,
              'lr': a.lr, 'ema': .998, 'seed': seed, 'encoder': 'convnext_tiny.fb_in22k_ft_in1k', 'bs': [a.bs_a, a.bs_b],
              'train_sources': sorted(tr_keys), 'inner_validation_sources': sorted(in_keys)}
    a.out.mkdir(parents=True, exist_ok=True)
    best, steps, hist, t0 = float('inf'), 0, [], time.monotonic()
    for ep, sel in enumerate(epochs):
        model.train()
        losses, batches = [], []
        data = {}
        for bk in 'AB':
            rows = [r for r in sel if r['bucket'] == bk]
            if rows:
                data[bk] = load(rows, bk)
                ids = torch.randperm(len(rows), device=dev, generator=gen)
                batches += [(bk, ids[i:i+bs[bk]]) for i in range(0, len(rows), bs[bk])]
        random.shuffle(batches)
        for bk, ids in batches:
            if len(ids) < 2:
                sch.step()
                continue
            x, t, w = data[bk]
            z = augment(torch.cat([x[ids].float()/255, t[ids].float()/255], 1), R.BUCKETS[bk]['inner'], gen)
            xb, tb = z[:, :4], z[:, 4:5].clamp(0, 1)
            xb = torch.cat([photometric(xb[:, :1], gen), xb[:, 1:]], 1)
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
                out = model(prep(xb))
            loss = loss_fn(out, tb, w[ids])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.)
            opt.step()
            sch.step()
            steps += 1
            losses.append(float(loss))
            d = min(.998, (1+steps)/(10+steps))
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.lerp_(pm, 1-d)
                for be, bm in zip(ema.buffers(), model.buffers()):
                    be.copy_(bm)
        v = validate(ema)
        hist.append({'epoch': ep, 'train_loss': float(np.mean(losses)) if losses else None, 'val_loss': v,
                     'sec': round(time.monotonic()-t0)})
        print(json.dumps(hist[-1]), flush=True)
        if v < best:
            best = v
            torch.save({'model': ema.state_dict(), 'config': {**config, 'best_epoch': ep}}, a.out/'best.pt')
    (a.out/'history.json').write_text(json.dumps(hist, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['bank', 'train'])
    ap.add_argument('--config', default='configs/final.yaml')
    ap.add_argument('--folds', type=Path, default=Path('configs/folds_full.json'))
    ap.add_argument('--data', type=Path, help='bank: build_verifier_data.py output (FZ)')
    ap.add_argument('--cnn', type=Path, help='bank: out-of-fold crop-CNN scores (FZ)')
    ap.add_argument('--bank', type=Path, help='train: bank directory')
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--trunc-p', type=float, default=.3)
    ap.add_argument('--alt-ratio', type=float, default=1.)
    ap.add_argument('--bs-a', type=int, default=48)
    ap.add_argument('--bs-b', type=int, default=16)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--smoke', action='store_true', help='bank: 2 frames per fold; train: 1 epoch')
    a = ap.parse_args()
    {'bank': cmd_bank, 'train': cmd_train}[a.cmd](a)


if __name__ == '__main__':
    main()
