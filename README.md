# Solar filament instance segmentation (Filament Segmentation Challenge 2026)

This is the code and model weights for our final entry to the Kaggle competition
[filament-segmentation-2026](https://www.kaggle.com/competitions/filament-segmentation-2026).

The task is instance segmentation of solar filaments in full-disk GONG H-alpha frames
(2048×2048). Every filament is one disjoint mask, submitted as a COCO RLE row.

Two pipelines are included, selected by config:
- **FZ_nz1** (default, `configs/fz_nz1.yaml`): the final pipeline. It is lite plus native-resolution
  small-filament specialists.
- **Lite** (`configs/f_lite.yaml`, also known as F_full_refit_nz1): our earlier submission, public
  leaderboard **0.39**. It has no specialists, which makes it about 2.3× faster on a CPU (104 s vs 238 s per frame).

**Reproduce the submissions:** open `notebooks/pipeline_demo.ipynb`, which is saved with its outputs.
It loads the weights stored in this repository, runs both pipelines on the test frames and checks each
`submission.csv` against the file we submitted. No retraining is needed.

## Method

**Base networks.** Two families of U-Nets, each trained as 5 calendar-month folds on all 707
labelled frames:
- ConvNeXt-T encoder: produces the segmentation
- ResNet34 encoder: provides a second opinion

Both families predict every frame at 1024² with 8-view dihedral test-time augmentation. The averaged
ConvNeXt foreground map is upsampled to 2048².

**Small-filament specialists (FZ only).** Downsampling to 1024 loses faint, thin, small filaments.
These are most of what the base networks miss.
- Each ConvNeXt fold network is fine-tuned on **native-resolution** 768-px crops. Half of the crops
  are centred on a small filament, and the loss puts extra weight on small filaments.
- The 5 specialists predict the full 2048 frame, with 8 views.
- Their mean map is averaged 50/50 with the base map.

**Candidates.** The (fused) map is thresholded at 0.4 and split into connected candidates of at least
150 px.

**Verifier.** It decides which candidates to keep:
- Inputs: 46 image-only features plus the score of a crop CNN.
- Model: a bagged gradient-boosting model that predicts u, the IoU a candidate is expected to have
  with an annotator's filament (0 if unmatched). A candidate is kept when u ≥ 0.25.
- It is trained on out-of-fold candidates, so it learns which kinds of detection the networks get
  wrong, such as faint fragments, limb features and doubled outlines.
- FZ has its own verifier, retrained on fused-map candidates.

**Outline.** Each kept candidate is re-outlined: among nested threshold levels of the same map, it
takes the outline with the highest expected IoU. Growth stays inside the candidate's Voronoi
territory, so instances remain disjoint.

## Quick start

```bash
git clone <this repo> && cd <repo>
python3.13 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt               # CPU build of PyTorch 2.11 (what we ran)
kaggle competitions download -c filament-segmentation-2026 -p data
unzip -q data/filament-segmentation-2026.zip -d data   # -> data/MAGFiLO_1.0_Kaggle_2026/
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/pipeline_demo.ipynb
# or just a submission file:
python -m filament_seg --output outputs/submission.csv               # FZ_nz1
python -m filament_seg --lite --output outputs/submission_lite.csv   # lite
```

Set `FILAMENT_DATA_ROOT=/path/to/MAGFiLO_1.0_Kaggle_2026` if the data lives elsewhere. For a quick look
on a CPU, set `FILAMENT_DEMO_MAX_FRAMES=3` before running the notebook.

- **GPU:** install the CUDA build of `torch==2.11.0`/`torchvision==0.26.0` instead of the `+cpu`
  pins. `inference.device: auto` then uses the GPU.
- **macOS:** drop the `+cpu` suffix.

## The two modes: score and runtime

**FZ_nz1 (default)**
- Networks per frame: 5 ConvNeXt + 5 ResNet34 at 1024², and 5 specialists at 2048², each with
  8 views. Then 5 crop CNNs and a GBM.
- Score: internal held-out PQ **+0.015** over lite (see Results); public leaderboard **0.39** (the same as lite at two decimals; one leaderboard step is about +0.012 internal PQ on only ~90 public images).
- CPU: **238 s per frame** (236 s of it is the networks), about **12 h** for the 180 test frames. This is the median of the 10 spot-check frames.
- GPU: not timed end to end for this repository. The logged GPU runs suggest about 15–20 s per frame,
  under an hour for the test set. Those logs recorded per network:
  - base ConvNeXt: about 1 s per frame on an H200
  - ResNet34: about 0.6 s per frame on an H200
  - specialist: about 1.7 s per frame on an A100

  All of these were measured with several jobs sharing the GPU, and include writing maps to disk.

**Lite (F_full_refit_nz1)**
- Networks per frame: 5 ConvNeXt + 5 ResNet34 at 1024², each with 8 views. Then 5 crop CNNs and a GBM.
- Score: public leaderboard **0.39**.
- CPU: **104 s per frame**, about **5.2 h** for the 180 test frames.
- GPU: about 8 s per frame by the same estimate (about 25 min for the test set); not timed end to end.

CPU figures were measured on 8 vCPUs (AMD EPYC Genoa), 16 GB RAM, no GPU, Python 3.13.5 and
PyTorch 2.11 (CPU build), with the machine not shared. They are medians per test frame.
- **Where the time goes.** Almost everything is network inference. The rest takes about 3 s per
  frame: candidates, features, crop CNNs, GBM, outline and RLE encoding.
- **Peak memory.** About 3–5 GB. The specialists run one view at a time at 2048².
- **Caching.** Results are cached per frame (`outputs/frames_*`), so an interrupted run resumes.

## Results

**Submitted files** (in `reference/`):
- **FZ_nz1:** `submission_FZ_nz1.csv`, sha256 `8f07cb44…3410`, 1169 masks
- **Lite:** `submission_F_full_refit_nz1.csv`, sha256 `df7e63f7…6c5e`, 1088 masks, public
  leaderboard 0.39
- In both, 177 of the 180 test frames have at least one mask.

**Byte-exact reproduction of the submitted files.** We fed the probability maps saved by the GPU runs
that produced each submission through this repository's code. It reproduces both files byte for
byte: FZ sha `8f07cb44…`, lite sha `df7e63f7…`, 180/180 frames each. `train/train_gbm.py`, run on the
same out-of-fold tables, rebuilds both GBM verifiers with identical predictions.

**End-to-end on a CPU.**
- **FZ, 10 test frames:** computed end to end, every 18th frame:
  - The same 93 candidates, the same keep/reject decision for every one of them, and the same 69 masks.
  - 1 of the 10 frames is bit-identical.
  - 29 masks differ by 1–4 pixels each, 45 pixels in total out of 126 k mask pixels (0.04%).
- **Lite, all 180 test frames:**
  - The same 1088 masks, with the same IDs and order, and the same keep/reject decision for every
    candidate.
  - 1046 masks are bit-identical.
  - 42 masks differ by 1–2 pixels each, 44 pixels in total out of 2.27 M mask pixels.

**Why the hash differs on a CPU.** The submissions' maps were computed on GPUs. The base networks ran
in TF32 there, and the specialists in bf16. CPU arithmetic differs at the level of one float16 step
(or one uint8 level for the specialists), which is enough to move single pixels across a threshold.

**Held-out labelled frames** (notebook section 6): 142 frames of fold 0 and 236 annotator records,
using only models that never saw these frames. These are single-fold pipelines, so they slightly
understate the 5-model ensembles. The metric is the host's PQ.

**FZ (default):**
- All candidates: PQ 0.436 (TP 1185, FP 900, FN 448)
- After the verifier: PQ 0.481
- After the verifier and the nz1 outline (final): PQ **0.483** (SQ 0.688, RQ 0.702)

**Lite:**
- All candidates: PQ 0.424 (TP 1118, FP 810, FN 515)
- After the verifier: PQ 0.461
- After the verifier and the nz1 outline (final): PQ **0.464** (SQ 0.681, RQ 0.682)

**FZ − lite: +0.019 PQ on these frames.** Recall of GT filaments under 1000 px rose from 0.526
to 0.595, and recall of larger ones from 0.711 to 0.732.

**Development protocol.**
- **Lite.** The recipe was trained on 531 frames and scored once on 80 untouched frames: PQ
  **0.457**. The previous pipeline scored 0.423 there (paired month-block 95% CI of the gain: +0.017
  to +0.055).
- **FZ.** It was pre-registered and assessed once on the same 80 frames: PQ **0.474**, a gain of
  **+0.015** over lite (95% CI +0.006 to +0.025). The gain is positive in all 14 leave-one-month-out
  subsets. It was also positive on the out-of-fold training frames (+0.014) and on the calibration
  frames (+0.011).
- **Where the FZ gain comes from.** Recall of small GT filaments (< 1000 px) rose from 0.53 to 0.60.

**Context.** On frames labelled by two annotators, one annotator scored against the other reaches a
PQ of about 0.34. Public leaderboard scores have run at about 0.86× our internal PQ.

## Weights

All weight files are under 95 MB, so no Git LFS is needed.

- **U-Nets** (15 networks): fp32 safetensors, 2 shards each:
  - ConvNeXt `cnx_fold*`
  - ResNet34 `r34_fold*`
  - specialists `spec_fold*`
- **Crop CNNs** `crop_cnn_fold*` (lite) and `crop_cnn_fz_fold*` (FZ):
  - They only run under bf16 autocast, which rounds their conv/linear weights to bf16 on every call.
  - Those tensors are therefore stored in bf16; this is lossless for the computation (the byte-exact
    checks above use these files).
  - BatchNorm tensors stay fp32.
- **GBM verifiers**, as gzip pickles:
  - `verifier_gbm.pkl.gz` (lite) and `verifier_gbm_fz.pkl.gz` (FZ)
  - the fold-0-excluded copies used by the held-out evaluation
- **`manifest.json`:** a SHA-256 of every reassembled model, checked on load.

We also tried storing the U-Nets in fp16. That changed the masks of a test frame that fp32
reproduces exactly, so the U-Nets stay fp32.

## Repository map

```
filament_seg/            the inference package (one module per stage)
  config.py              YAML config loading; all paths relative to the repo root
  data.py                image discovery, preprocessing, ground-truth loading
  models.py              ConvNeXt-T U-Net, ResNet34 U-Net, crop CNN; sharded weight loading + hash check
  inference.py           8-view TTA, fold ensembling, specialists at 2048 and map fusion
  candidates.py          connected-component candidates (fg >= 0.4, area >= 150, holes filled)
  features.py            the 46 verifier features
  verifier.py            crop extraction, crop-CNN scores, bagged GBM, keep decision
  postprocess.py         configurable post-processing steps: nz1 outline, drop rejected
  submission.py          RLE encoding, CSV writing and validation
  metrics.py             host's PQ, IoU/Dice distributions, 1:n / n:1 relations
  pipeline.py            frame-level pipeline + resumable directory runner
  viz.py                 plots and overlays
  __main__.py            command line: python -m filament_seg [--lite]
configs/
  fz_nz1.yaml            every setting of the default FZ_nz1 pipeline
  f_lite.yaml            the lite pipeline (F_full_refit_nz1)
  folds_full.json        calendar-month folds of the 707 labelled frames
weights/                 44 weight files + manifest.json, 2.08 GB, all < 95 MB (see Weights)
notebooks/pipeline_demo.ipynb   the end-to-end demo (executed)
train/                   scripts that produced the weights (see train/README.md)
reference/               the two submitted CSVs (for the reproduction checks)
tests/                   fast data-free unit tests
```

## Data and rules we followed

- **Training data.** Only the official competition training images and their annotations
  (707 frames, 1154 annotator records).
  - No external labels.
  - No other MAGFiLO releases.
  - No additional GONG frames in the final models. Extra GONG frames and pseudo-labels were tried
    during development and dropped because they lowered the score.
- **Inference uses only the test image itself.** Nothing else is read at inference time:
  - no file-name date or site
  - no metadata
  - no neighbouring frames
  - no test-time training
- **The annotator structure is used only during training.** One frame has several annotators; the
  soft targets average them, and the verifier's targets are averaged over annotators.
- **Leaderboard feedback did not select the pipeline.** Model selection used held-out labelled
  months, and τ, the thresholds and the outline rule were frozen before any test export.

## Citation

The competition data is a subset of MAGFiLO:

> Ahmadzadeh, A., Adhyapak, R., Chaurasiya, K., et al. *A dataset of manually annotated filaments
> from H-alpha observations.* Scientific Data 11, 1031 (2024). https://doi.org/10.1038/s41597-024-03876-y

```bibtex
@article{ahmadzadeh2024magfilo,
  title   = {A dataset of manually annotated filaments from H-alpha observations},
  author  = {Ahmadzadeh, Azim and Adhyapak, Rohan and Chaurasiya, Kartik and Nagubandi, Laxmi Alekhya and
             Aparna, V and Martens, Petrus C and Pevtsov, Alexei and Bertello, Luca and Pevtsov, Alexander and
             others},
  journal = {Scientific Data},
  volume  = {11},
  pages   = {1031},
  year    = {2024},
  doi     = {10.1038/s41597-024-03876-y}
}
```

The H-alpha frames come from the NSO Global Oscillation Network Group (GONG).

The expected-IoU outline follows S. Nowozin, *Optimal Decisions from Probabilistic Models: the
Intersection-over-Union Case*, CVPR 2014.

## License

MIT (see `LICENSE`). The competition data is not included and keeps its own terms.
