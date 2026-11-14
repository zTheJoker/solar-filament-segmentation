# Solar filament instance segmentation (Filament Segmentation Challenge 2026)

This is the code and model weights for our final entry to the Kaggle competition
[filament-segmentation-2026](https://www.kaggle.com/competitions/filament-segmentation-2026).

The task is instance segmentation of solar filaments in full-disk GONG H-alpha frames
(2048×2048). Every filament is one disjoint mask, submitted as a COCO RLE row.

**Our final submission is the default pipeline, `configs/final.yaml`** (Kaggle submission 58253542, public
leaderboard **0.40**): FZ_nz1 plus context networks, detector votes, a keep model, an instance refiner and a
sub-pixel shift.

Two earlier versions are kept because the final pipeline is built on them and the ablation in [METHODS.md](METHODS.md)
compares against them; neither is our final submission:
- **FZ_nz1** (`configs/fz_nz1.yaml`, public 0.39): dense U-Nets with native-resolution small-filament specialists, a
  verifier and the expected-IoU outline. Its candidates and verifier are the first half of the final pipeline.
- **Lite** (`configs/f_lite.yaml`, also known as F_full_refit_nz1, public 0.39): FZ_nz1 without the specialists.

**Reproduce the final submission:** open `notebooks/pipeline_demo.ipynb`, which is saved with its outputs. It loads the
weights stored in this repository, runs the final pipeline on the 180 test frames and checks the result against the
file we submitted. No retraining is needed. The methods, the validation protocol and what did not work are described
in [METHODS.md](METHODS.md).

## Method

**Dense networks (20).** Every network family is trained as 5 calendar-month folds on all 707 labelled frames; every
network averages 8 dihedral views.
- 5 ConvNeXt-T U-Nets at 1024²: the base segmentation map.
- 5 small-filament specialists: the same U-Nets fine-tuned on native-resolution crops with extra weight on small
  filaments, run on the full 2048² frame. Their map is averaged 50/50 with the upsampled base map: the **fused map**.
- 5 ResNet34 U-Nets at 1024²: a second opinion, used only through six vote features.
- 5 context U-Nets: ConvNeXt-T U-Nets whose input is the frame plus the maps of the ConvNeXt and ResNet34 networks of
  the same fold. Used only through six vote features.

**Candidates.** The fused map is thresholded at 0.4 and split into connected candidates of at least 150 px.

**FZ verifier.** 46 image-only features plus the score of a crop CNN (5 ResNet18) go into a bagged gradient-boosting
model that predicts u_FZ, the IoU a candidate is expected to have with an annotator's filament (0 if unmatched).

**Expected-IoU outline (nz1).** Every candidate is re-outlined: among nested threshold levels of the fused map it
takes the outline with the highest expected IoU, growing only inside its own Voronoi territory, so the outlines stay
disjoint.

**Detector votes.** Twenty YOLO26m-seg detectors (4 training seeds × 5 folds) run on the native frame. Their
instances are never output; six agreement columns per candidate (best IoU with a detector instance, its confidence,
coverage, …), measured on the nz1 outline and averaged over the detectors, tell the keep model how a different kind of
model sees the candidate.

**Keep model.** A second bagged GBM on 59 columns (the 46 features, the crop-CNN score, 6 context votes and 6
detector columns) predicts u; a candidate is kept when u ≥ 0.25 (keeping raises PQ only when u is above about PQ/2).

**Refiner.** Five instance refiners (ConvNeXt-T U-Nets on rotated native-resolution crops of each candidate, with the
image, the fused map of their own fold, the candidate and its neighbours as input) re-draw the kept candidates that
the FZ verifier is least sure of (u_FZ ≤ 0.55): those are mostly filaments the dense maps cut short.

**Sub-pixel shift and paste-back.** The annotators' drawings sit about 0.23 px above our flip-symmetric masks. The
refined probability and, for the other kept candidates, the fused map the nz1 outline is cut from are moved by
(−0.23, +0.05) px before thresholding. Overlaps go to the higher refined probability; every filament is one disjoint
mask.

## Quick start

```bash
git clone <this repo> && cd <repo>
python3.13 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt               # CPU build of PyTorch 2.11 (see below for CUDA)
kaggle competitions download -c filament-segmentation-2026 -p data
unzip -q data/filament-segmentation-2026.zip -d data   # -> data/MAGFiLO_1.0_Kaggle_2026/
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/pipeline_demo.ipynb
# or just a submission file:
python -m filament_seg --output outputs/submission.csv                                        # final submission
python -m filament_seg --fz --output outputs/submission_fz.csv                                # FZ_nz1
python -m filament_seg --lite --output outputs/submission_lite.csv                            # lite
```

Set `FILAMENT_DATA_ROOT=/path/to/MAGFiLO_1.0_Kaggle_2026` if the data lives elsewhere. `FILAMENT_DEMO_MAX_FRAMES=3`
runs the notebook on three test frames.

- **GPU (recommended for the final pipeline):** `pip install -r requirements-cuda.txt` instead (the same versions with
  the CUDA 12.8 builds of PyTorch; the environment of the saved notebook run). `inference.device: auto` then uses the
  GPU.
- **With uv:** `uv pip install --index-strategy unsafe-best-match -r requirements.txt`.
- **macOS:** drop the `+cpu` suffix.
- `ultralytics-opencv-headless` is the Ultralytics package built against OpenCV-headless; it runs the detectors with
  telemetry switched off.

## Run time and hardware

**Final pipeline, measured end to end** (the saved notebook run): one NVIDIA RTX 5090 (32 GB), 32 vCPUs, Python
3.13.5, PyTorch 2.11.0 with CUDA 12.8.
- 180 test frames in **76 minutes**, a median of **24.7 s per frame**:
  - segmenters, voters and specialists (15 U-Nets × 8 views): 9.0 s
  - detectors: 13.0 s, most of it the CPU-side computation of the agreement columns from the 20 models' masks
  - context networks 0.9 s, refiner 0.5 s, candidates, features, verifier, keep model, paste-back and encoding 1.3 s
- Loading the 50 networks and 2 GBMs: 8.5 s. The whole notebook (test set plus the 142 held-out frames of section 6):
  87 minutes.
- Peak GPU memory about 18 GB with every model resident.

**FZ_nz1 and lite on a CPU** (measured for the earlier versions of this repository, 8 vCPUs AMD EPYC Genoa, PyTorch 2.11
CPU build): 238 s and 104 s per frame. The final pipeline was not timed end to end on a CPU; it adds 5 context networks
at 1024², 20 detectors at 1536 and the refiners to FZ_nz1's networks.
- **Caching.** Results are cached per frame (`outputs/frames_*`), so an interrupted run resumes.

## Results

**Reference files** (in `reference/`):
- **Final submission:** `submission_final.csv`, sha256 `3b136cbe…7195987`, 1187 masks on 177 images, public
  leaderboard 0.40
- Earlier version **FZ_nz1:** `submission_FZ_nz1.csv`, sha256 `8f07cb44…3410`, 1169 masks, public 0.39
- Earlier version **Lite:** `submission_F_full_refit_nz1.csv`, sha256 `df7e63f7…6c5e`, 1088 masks, public 0.39

**What each config reproduces.**

Two checks, for each reference file:
1. **From the stored maps: byte-identical.** We fed the intermediate outputs saved by the GPU runs that produced
   each submission (dense maps, context maps, detector instances, refined probabilities) through this repository's
   code. All three files come out byte for byte: final `3b136cbe…`, FZ_nz1 `8f07cb44…`, lite `df7e63f7…`. Every intermediate stage matches too: the context votes, the detector columns, the keep
   model's scores (max |Δu| = 0), the nz1 outlines, the shifted outlines and the final masks. `train/train_gbm.py`,
   run on the same out-of-fold tables, rebuilds both keep models with identical test scores.
2. **End to end on a GPU: the same decisions, a few pixels apart.** The submissions' intermediate outputs were
   computed on several GPU types with other PyTorch versions. Recomputing them on another GPU changes floating-point
   details (single float16 steps in the maps, one uint8 level in a few refined probabilities), which moves single
   pixels across a threshold. So the end-to-end files are not byte-identical. We measured how far they are:
   - **Final** (`configs/final.yaml`, notebook run, RTX 5090): the same keep decision for every candidate and the
     same number of masks in every frame (1187); 41 of 180 frames bit-identical; 278 masks differ by 620 pixels in
     total (0.03 % of 2.29 M mask pixels; the largest change is 70 pixels on one mask, IoU 0.94). One frame gains a
     low-scored candidate, which shifts the row order of two masks.
   - **FZ_nz1**, as a by-product of the same runs: 1168 of 1169 masks on the RTX 5090 (one candidate's u_FZ crosses
     0.25), 234 masks differ by 301 pixels.
   - Stage by stage on the GPU, each stage fed the stored inputs of that stage (3 frames): context maps within one
     float16 step (vote features within 0.0014); refined probabilities within one uint8 level (2 of 2.9 M pixels
     change side of the 0.60 threshold); detectors: 19 of 20 models give the same number of instances, confidences
     within 0.0013.

**Held-out results** (pre-registered single runs on 80 assessment frames that no design choice used; every model
trained on 531 other labelled frames; host's PQ pooled over 140 annotator records; details in METHODS.md):
- FZ_nz1: PQ 0.4735 (673 matches, 275 false positives, 335 misses)
- Final: PQ **0.4806** (666 matches, 241 false positives, 342 misses; SQ 0.691, RQ 0.696)
- The final pipeline keeps fewer false positives and splits fewer filaments; its gain over FZ_nz1 on these 80 frames
  is within their noise, and both show 0.39–0.40 on the public leaderboard.

**Held-out frames in the notebook** (section 6): the 142 frames of fold 0 and 236 annotator records, using only
the fold-0 networks, detectors and refiner and GBMs refit without fold 0, so no model saw these frames. These are
single-fold pipelines, which understate the 5-model ensembles (the refiner's gain is an ensemble effect).
- FZ_nz1: PQ 0.4820 (TP 1101, FP 408, FN 532; SQ 0.688, RQ 0.701)
- Final: PQ **0.4867** (TP 1085, FP 363, FN 548; SQ 0.691, RQ 0.704), +0.0047
- The final pipeline drops 45 false positives for 16 lost matches, and splits fewer filaments (GT filaments covered
  by two or more predictions: 51 → 36).

**Context.** On frames labelled by two annotators, one annotator scored against the other reaches a PQ of about
0.34. Public leaderboard scores have run at about 0.84–0.86× our internal PQ.

## Weights

All weight files are under 95 MB, so no Git LFS is needed. `weights/manifest.json` holds a SHA-256 of every file
or model; the networks (hash of the reassembled state dict) and the detectors (hash of the file) are checked on load.

- **Dense U-Nets** (`cnx_fold*`, `spec_fold*`, `r34_fold*`): fp32 safetensors, 2 shards each. We tried fp16 storage;
  it changed the masks of a test frame that fp32 reproduces exactly, so they stay fp32.
- **Crop CNNs** (`crop_cnn_fold*`, `crop_cnn_fz_fold*`), **context networks** (`l2_fold*`) and **refiners**
  (`refiner_fold*`): they only run under bf16 autocast, which rounds their convolution and linear weights to bf16 on
  every call, so those tensors are stored in bf16 (the computation is identical; checked on CPU and GPU). The other
  tensors stay fp32.
- **Detectors** (`yolo_s{seed}_f{fold}.pt`): the Ultralytics checkpoints as trained (fp16).
- **GBMs** (gzip pickles, scikit-learn 1.9.1): `verifier_gbm.pkl.gz` (lite), `verifier_gbm_fz.pkl.gz` (FZ verifier),
  `keep_gbm_y4.pkl.gz` and `keep_gbm_dv.pkl.gz` (keep models of the two final configs), and fold-0-excluded copies
  for the notebook's held-out evaluation.

## Repository map

```
filament_seg/            the inference package (one module per stage)
  config.py              YAML config loading; all paths relative to the repo root
  data.py                image discovery, preprocessing, ground-truth loading
  models.py              U-Nets, crop CNN, refiner; sharded weight loading + hash check
  inference.py           8-view TTA, fold ensembling, specialists at 2048, map fusion, per-fold maps
  candidates.py          connected-component candidates (fg >= 0.4, area >= 150, holes filled)
  features.py            the 46 verifier features (incl. the six vote features)
  verifier.py            crop extraction, crop-CNN scores, bagged GBM, keep decision
  context.py             context networks -> six vote features
  detector.py            YOLO detectors -> six agreement columns
  keep.py                keep model (59 columns)
  refiner.py             rotated native crops, refiner networks, refined probabilities
  pasteback.py           sub-pixel shift, re-drawn and re-cut masks, overlap resolution
  postprocess.py         configurable post-processing steps: nz1 outline, drop rejected
  submission.py          RLE encoding, CSV writing and validation
  metrics.py             host's PQ, IoU/Dice distributions, 1:n / n:1 relations
  pipeline.py            frame-level pipeline + resumable directory runner
  viz.py                 plots and overlays
  __main__.py            command line: python -m filament_seg [--config ... | --fz | --lite]
configs/
  final.yaml             every setting of the final pipeline
  fz_nz1.yaml            FZ_nz1
  f_lite.yaml            lite (F_full_refit_nz1)
  folds_full.json        calendar-month folds of the 707 labelled frames
weights/                 77 weight files + manifest.json, 3.89 GB, all < 95 MB (see Weights)
notebooks/pipeline_demo.ipynb   the end-to-end demo (executed)
train/                   scripts that produced the weights (see train/README.md)
reference/               the final submission and the two earlier versions' CSVs (for the reproduction checks)
requirements.txt         pinned versions, CPU build of PyTorch; requirements-cuda.txt: the same with CUDA 12.8 builds
tests/                   fast data-free unit tests
METHODS.md               methods, validation protocol, ablations, limitations
```

## Data and rules we followed

- **Training data.** Only the official competition training images and their annotations
  (707 frames, 1154 annotator records).
  - No external labels.
  - No other MAGFiLO releases.
  - No additional GONG frames in the final models. Extra GONG frames and pseudo-labels were tried
    during development and dropped because they lowered the score.
  - Pretrained starting points: ImageNet weights of the encoders (timm, torchvision) and the official COCO weights of
    YOLO26m-seg.
- **Inference uses only the test image itself.** Nothing else is read at inference time:
  - no file-name date or site
  - no metadata
  - no neighbouring frames
  - no test-time training
- **The annotator structure is used only during training.** One frame has several annotators; the
  soft targets average them, the detectors see one sample per annotator record, and the keep model's targets are
  averaged over annotators.
- **Leaderboard feedback did not select the pipeline.** Every stage was adopted by a rule written down before the
  held-out frames were scored (METHODS.md, section 2), and the thresholds and the shift were frozen before any test
  export.

## Third-party components

Licences as declared in the package metadata of the pinned versions (`requirements.txt`) and by the weight
publishers; please check them before redistributing.
- **Ultralytics** (`ultralytics-opencv-headless==8.4.210`, detector training and inference) and **ultralytics-thop**:
  AGPL-3.0. The official COCO weights `yolo26m-seg.pt` the detectors were fine-tuned from are published by Ultralytics
  under AGPL-3.0, so the detector weights in `weights/yolo_*.pt` are derived from AGPL-3.0 material.
- **PyTorch** 2.11.0: BSD-3-Clause. **torchvision** 0.26.0: BSD-3-Clause (its ImageNet ResNet34 weights initialised
  the ResNet34 U-Nets).
- **timm** 1.0.31: Apache-2.0. The encoders were initialised from the timm weights `convnext_tiny.fb_in22k_ft_in1k`
  (ConvNeXt release by Meta, MIT licence) and `resnet18.a1_in1k` (timm, Apache-2.0).
- **scikit-learn** 1.9.1 (FZ verifier and keep model): BSD-3-Clause. **NumPy** 2.5.4 and **SciPy** 1.18.1:
  BSD-3-Clause. **threadpoolctl**: BSD-3-Clause. **scikit-image**: BSD-3-Clause.
- **pycocotools** 2.0.11: BSD-2-Clause. **OpenCV** (`opencv-python-headless` 5.0.0.93): Apache-2.0.
  **safetensors** 0.8.0: Apache-2.0. **Pillow** 13.0.0: MIT-CMU. **PyYAML**: MIT. **Matplotlib**: PSF-based licence.
- The competition data (MAGFiLO, GONG H-alpha frames) is not included and keeps its own terms.

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

Copyright (C) 2026 Eli Rubinstein. This repository is licensed under the GNU Affero General Public License v3.0
(AGPL-3.0, see `LICENSE`).

The final pipeline trains and runs its detectors with the Ultralytics library and ships detector weights fine-tuned
from Ultralytics' COCO weights, both AGPL-3.0, so a repository that distributes them has to be AGPL-3.0 as well.
Before this change the code of the other stages was released under the MIT licence.

The competition data is not included and keeps its own terms.
