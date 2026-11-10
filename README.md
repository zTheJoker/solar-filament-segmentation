# Solar filament instance segmentation (Filament Segmentation Challenge 2026)

This is the code and model weights for our final submission to the Kaggle competition
[filament-segmentation-2026](https://www.kaggle.com/competitions/filament-segmentation-2026).
The submission, pipeline **F_full_refit_nz1**, scored **0.39** on the public leaderboard (PQ).

The task is instance segmentation of solar filaments in full-disk GONG H-alpha frames
(2048×2048). Every filament is one disjoint mask, submitted as a COCO RLE row.

**Reproduce the submission:** open `notebooks/pipeline_demo.ipynb`. It loads the weights stored in
this repository, runs the whole pipeline on the test frames and checks the resulting
`submission.csv` against the file we submitted. No retraining is needed. The notebook is saved with
its outputs.

## Method in one paragraph

Two families of U-Nets, each trained as 5 calendar-month folds on all 707 labelled frames:
- ConvNeXt-T encoder: produces the segmentation
- ResNet34 encoder: provides a second opinion

Both families predict every frame at 1024² with 8-view dihedral test-time augmentation. The averaged
ConvNeXt foreground map is upsampled to 2048², thresholded at 0.4 and split into connected
candidates of at least 150 px.

A **verifier** then decides which candidates to keep. Its inputs are 46 image-only features plus the
score of a crop CNN, and it is a bagged gradient-boosting model. It predicts u, the IoU a candidate
is expected to have with an annotator's filament (0 if unmatched), and keeps the candidate when
u ≥ 0.25. Because the verifier is trained on out-of-fold candidates, it learns which kinds of
detection the networks get wrong, such as faint fragments, limb features and doubled outlines.

Each kept candidate is finally re-outlined: among nested threshold levels of the same map, it takes
the outline with the highest expected IoU. Growth stays inside the candidate's Voronoi territory,
so instances remain disjoint.

## Quick start

```bash
git clone <this repo> && cd <repo>
python3.13 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt               # CPU build of PyTorch 2.11 (what we ran)
kaggle competitions download -c filament-segmentation-2026 -p data
unzip -q data/filament-segmentation-2026.zip -d data   # -> data/MAGFiLO_1.0_Kaggle_2026/
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/pipeline_demo.ipynb
# or just the submission:
python -m filament_seg --output outputs/submission.csv
```

Set `FILAMENT_DATA_ROOT=/path/to/MAGFiLO_1.0_Kaggle_2026` if the data lives elsewhere.

- **GPU:** install the CUDA build of `torch==2.11.0`/`torchvision==0.26.0` instead of the `+cpu`
  pins. `inference.device: auto` then uses the GPU.
- **macOS:** drop the `+cpu` suffix.

## Runtime and hardware (measured)

Measured on a CPU-only machine:
- 8 vCPUs (AMD EPYC Genoa), 16 GB RAM, no GPU
- Python 3.13.5, PyTorch 2.11 CPU build

Per test frame (median of 47 frames, 8 threads, machine not shared):
- **total: 104 s**
  - networks: 102 s (10 U-Nets × 8 TTA views at 1024²)
  - everything after them: about 3 s (candidates, features, crop CNNs, GBM, outline and RLE encoding)

Other figures:
- **Whole test set (180 frames):** about 5.2 h on this CPU. The work is almost entirely network
  inference, so a GPU shortens it to a small fraction of that; we did not time a GPU run for this
  repository.
- **Loading all 15 networks (10 U-Nets, 5 crop CNNs) and the GBM:** about 8 s.
- **Peak memory:** 2–5 GB.
- **Held-out evaluation in the notebook:** one fold model per family, about 40 s per frame.
- **Cache:** results are cached per frame (`outputs/frames_*`), so an interrupted run resumes.

## Results

**Leaderboard.** The submitted file scored **0.39** on the public leaderboard (PQ). It is in
`reference/submission_F_full_refit_nz1.csv` (sha256 `df7e63f7…6c5e`, 1088 masks, 177 of the 180
test frames have at least one).

**Reproduction from this repository** (`notebooks/pipeline_demo.ipynb`, CPU):
- **Masks.** We get the same 1088 masks, with the same IDs and order, and the same keep/reject
  decision for every candidate. The 1088 masks cover 2.27 M pixels in total:
  - 1046 masks (144 of 180 frames) are bit-identical
  - 42 masks differ by 1–2 pixels each, 44 pixels in total, about 0.002% of all mask pixels
- **Why the file hash differs.** The submitted maps were computed on an H200 GPU, and cuDNN
  convolutions there use TF32 arithmetic. CPU float32 maps differ from them by at most about 0.0005
  on the frames we checked, which is one float16 step. That is enough to move single pixels across the 0.4 threshold or the
  outline levels.
- **The post-network code is exact.** Fed the probability maps that the GPU run saved, this
  repository's code reproduces the submitted file byte for byte: sha256 `df7e63f7…`, 180/180 frames.
- **Verifier.** `train/train_gbm.py`, run on the same out-of-fold table, rebuilds the submitted GBM
  with identical predictions.
- **Why the weights are fp32.** We also tried storing the weights in fp16, which would halve the
  repository. Rounding them to fp16 changed the masks of a test frame that fp32 reproduces exactly,
  so we kept fp32 and split each network into shards under 95 MB.

**Held-out labelled frames** (notebook section 6): 142 frames of fold 0 and 236 annotator records,
using only models that never saw these frames. This is a single-fold pipeline, so it slightly
understates the 5-model ensemble. The metric is the host's PQ.

- All candidates: PQ 0.424 (TP 1118, FP 810, FN 515)
- After the verifier: PQ 0.461 (TP 1028, FP 367, FN 605)
- After the verifier and the nz1 outline (final): PQ **0.464** (SQ 0.681, RQ 0.682)

**Development protocol.** The same recipe was trained on 531 frames and scored once on 80
untouched frames, giving PQ **0.457**. The previous pipeline scored 0.423 on those frames
(paired month-block 95% CI of the gain: +0.017 to +0.055).

**Context.** On frames labelled by two annotators, one annotator scored against the other reaches
a PQ of about 0.34. Public leaderboard scores have run at about 0.86× our internal PQ.

## Repository map

```
filament_seg/            the inference package (one module per stage)
  config.py              YAML config loading; all paths relative to the repo root
  data.py                image discovery, preprocessing, ground-truth loading
  models.py              ConvNeXt-T U-Net, ResNet34 U-Net, crop CNN; sharded weight loading + hash check
  inference.py           8-view TTA, fold ensembling, upsampling to 2048
  candidates.py          connected-component candidates (fg >= 0.4, area >= 150, holes filled)
  features.py            the 46 verifier features
  verifier.py            crop extraction, crop-CNN scores, bagged GBM, keep decision
  postprocess.py         configurable post-processing steps: nz1 outline, drop rejected
  submission.py          RLE encoding, CSV writing and validation
  metrics.py             host's PQ, IoU/Dice distributions, 1:n / n:1 relations
  pipeline.py            frame-level pipeline + resumable directory runner
  viz.py                 plots and overlays
  __main__.py            command line: python -m filament_seg
configs/
  f_full_refit_nz1.yaml  every setting of the submitted pipeline
  folds_full.json        calendar-month folds of the 707 labelled frames
weights/                 28 files, 1.47 GB (all < 95 MB, no Git LFS): 10 U-Nets as fp32 safetensors
                         (2 shards each), 5 crop CNNs, the GBM verifier (+ fold-0-excluded copy),
                         manifest.json with per-model SHA-256
notebooks/pipeline_demo.ipynb   the end-to-end demo (executed)
train/                   scripts that produced the weights (see train/README.md)
reference/               the CSV we submitted (for the reproduction check)
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
