# Training scripts

These scripts reproduce the weights in `../weights/`. You do not need them to reproduce the
submission: the notebook loads the stored weights.

They are cleaned-up versions of the scripts we actually ran, and the recipes and hyperparameters are
the same. Because of GPU nondeterminism (cuDNN, bf16 autocast, data-loader order), a retrain gives
equivalent but not bit-identical weights.

Hardware for the released runs:
- **U-Nets:** one NVIDIA H200 (RunPod), with the five folds of each family trained concurrently.
- **Specialists:** one NVIDIA A100.

Approximate time per fold:
- ConvNeXt U-Net: 1.3–1.5 h
- ResNet34 U-Net: about 1 h
- Specialist fine-tune: about 15 min
- Crop CNN: about 1 min

Run everything from the repository root:

```bash
# 0. caches: normalised 768/1024 inputs + per-annotator soft targets; native-2048 caches for the specialists (CPU)
python train/prepare_data.py --out work/prepared --sides 768 1024
python train/train_specialist.py prep --out work/native

# 1. segmenters, voters and specialists, one call per fold k = 0..4 (GPU)
python train/train_convnext_unet.py --cache work/prepared/1024 --fold $k --out work/runs/cnx_fold$k
python train/train_r34_unet.py --cache768 work/prepared/768 --cache1024 work/prepared/1024 --fold $k --out work/runs/r34_fold$k
python train/train_specialist.py train --cache work/native --fold $k --init work/runs/cnx_fold$k/best.pt --out work/runs/spec_fold$k

# 2-4 run once per mode: FZ (default) shown; for lite use --config configs/f_lite.yaml and drop the _fz suffixes
# 2. out-of-fold candidates of all 707 labelled frames: features, targets, crops (FZ: on fused maps)
python train/build_verifier_data.py --config configs/fz_nz1.yaml --runs work/runs --out work/verifier_data_fz.npz

# 3. crop CNNs (5 folds) and their out-of-fold scores
python train/train_crop_cnn.py --data work/verifier_data_fz.npz --out work/runs/crop_cnn_fz

# 4. bagged GBM verifier, plus the fold-0-excluded copy used for the notebook's held-out evaluation
python train/train_gbm.py --data work/verifier_data_fz.npz --cnn work/runs/crop_cnn_fz/cnn_oof_scores.npz --out work/runs/verifier_gbm_fz.pkl
python train/train_gbm.py --data work/verifier_data_fz.npz --cnn work/runs/crop_cnn_fz/cnn_oof_scores.npz --exclude-fold 0 \
       --out work/runs/verifier_gbm_fz_without_fold0.pkl

# 5. weights/ (U-Nets fp32 shards < 95 MB, crop-CNN conv/linear tensors bf16, GBMs gzip; hashes in manifest.json)
python train/export_weights.py --cnx work/runs/cnx_fold{0,1,2,3,4}/best.pt --r34 work/runs/r34_fold{0,1,2,3,4}/best.pt \
       --crop-cnn work/runs/crop_cnn/cnn_fold{0,1,2,3,4}.pt --gbm work/runs/verifier_gbm.pkl \
       --spec work/runs/spec_fold{0,1,2,3,4}/best.pt --crop-cnn-fz work/runs/crop_cnn_fz/cnn_fold{0,1,2,3,4}.pt \
       --gbm-fz work/runs/verifier_gbm_fz.pkl \
       --extra-gbm verifier_gbm_without_fold0.pkl.gz=work/runs/verifier_gbm_without_fold0.pkl \
                   verifier_gbm_fz_without_fold0.pkl.gz=work/runs/verifier_gbm_fz_without_fold0.pkl --out weights
```

Every script accepts `--smoke`, which runs it on a few frames for one epoch. This checks that the
chain works on a CPU machine; it does not produce useful weights.

## Folds (`configs/folds_full.json`)

All 707 labelled frames are grouped by calendar month and assigned to 5 folds, balanced greedily
by frame count; each fold holds out 141 or 142 frames.

Within each fold:
- Training frames within ±4 days of a held-out frame are purged, because filaments persist for days.
- About 10% of the remaining frames, taken as whole months, form an inner slice that selects the
  checkpoint.

`common.make_folds` regenerates the file exactly from the list of labelled frames.

Annotator records of the same frame always stay together.

## What each stage learns from

- **Specialists** start from the same fold's ConvNeXt network and train on that fold's training
  frames only, using native-resolution crops of the official annotations.
- **U-Nets** learn from the official training annotations only. No external labels or extra
  frames were used. GONG frames and pseudo-labels were tried in development and both lowered the
  score.
- **The verifier** learns from out-of-fold candidates. Every candidate comes from a network that
  never saw its frame, so the verifier sees test-like errors.
- **τ = 0.25 and the candidate settings (0.4 threshold, 150 px).** Both were chosen in an earlier
  round that trained on 531 frames only. That round held out:
  - 71 calibration frames, used for every choice
  - 80 assessment frames, scored exactly once (PQ 0.4567 vs 0.4231 for the previous pipeline)
- **The nz1 outline.** Its γ = 1 is the theory value, not a tuned one. It was checked on the
  calibration frames (PQ 0.4494 → 0.4520) and on the out-of-fold training frames.
- **FZ.** The specialist recipe and the 50/50 fusion were pre-registered and assessed once on the
  same 80 frames, PQ 0.474 vs 0.458 for lite. That run trained on the 531 frames.
- **The full refits** (all 707 frames, the released weights) reuse all of these settings unchanged.
