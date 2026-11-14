# Methods

This document describes the final pipeline of our entry to the Kaggle competition
[filament-segmentation-2026](https://www.kaggle.com/competitions/filament-segmentation-2026), how each stage was trained
and validated, what we measured, what did not work, and which public work it builds on. The README describes how to run
it. All numbers are panoptic quality (PQ) as computed by the host's scorer: per annotator record, pooled over records.

## 1. The task and the target we chose

- Input: one 2048 × 2048 grayscale GONG H-alpha frame. Output: disjoint filament masks, one COCO RLE row per filament.
- The labels are noisy. 296 of the 707 labelled frames have two or three annotator records. Scoring every ordered pair
  of annotators against each other with the competition metric gives PQ 0.340.
- The metric matches a prediction to an annotated filament at IoU > 0.5, separately for each annotator record, and pools
  over records: PQ = (sum of matched IoU) / (TP + FP/2 + FN/2). A predicted mask is a true positive for the annotators who
  drew a matching filament and a false positive for the others.
- Two consequences shape the pipeline.
  - We train the dense networks on the per-pixel mean of the annotators' masks (a soft target), because the best outline
    maximises expected IoU over annotators.
  - For a candidate mask, let u be its matched IoU averaged over the image's annotator records (0 for a record it does
    not match). Adding the candidate adds 1/2 to the denominator for every record and u to the numerator, so it raises
    PQ exactly when u > PQ/2. We therefore predict u for every candidate and keep it if u ≥ 0.25.

## 2. Data, splits and validation protocol

- Data: 707 labelled frames ("sources") with 1154 annotator records and 8199 instances; 180 unlabelled test frames. No
  external labels were used anywhere.
- Splits are by source and calendar month with a time purge, because neighbouring days look alike.
  - Outer split: 531 TRAIN sources, 151 validation sources, 25 buffer sources (7-day purge).
  - The validation sources are split into 71 calibration sources (15 months) and 80 assessment sources (14 months).
  - TRAIN is split into five calendar-month folds (106–107 held-out sources each); training sources within ±4 days of a
    held-out source are dropped from that fold.
- Cross-fitting. Every learned stage uses the same five folds.
  - A TRAIN source is only ever processed by the fold model that did not train on it. We call these predictions
    TRAIN-OOF (out of fold).
  - Calibration, assessment and test sources are processed by all five fold models, and the outputs are averaged.
  - Each stage after the dense networks is trained on TRAIN-OOF inputs, so it sees the same kind of input at training
    and at test time. Trainers and inference code assert that no model reads a source it trained on.
- Three evaluation sets.
  - TRAIN-OOF (531 sources) and calibration (71 sources) were used for every design choice.
  - Assessment (80 sources) was used only in pre-registered single runs: code, score tables and the masks of every
    candidate arm were fixed and hashed, the decision rule was written down, and then the assessment labels were decoded
    once by a command that refuses to run twice. The round-1, round-2 and round-3 pre-registrations have sha256 prefixes
    d3694475, 66bfa094 and e59a662b.
  - The assessment set has served seven such single runs over the project. Its intervals are conditional on the choices
    made before each run.
- Intervals are 95% paired bootstrap intervals that resample whole calendar months (for assessment: 14 months, 80
  sources, 140 annotator records, 20,000 resamples).
- Two "worlds". All numbers in this document come from models trained on TRAIN labels only (round-1 folds). For the test
  submission every stage was refit with the same recipe and settings on all 707 labelled sources (five new month folds
  of 141–142 held-out sources). These refits have no internal score, because no labelled source is left out of them.

## 3. The pipeline, stage by stage

### 3.1 Preprocessing

- In: the 2048² frame. Out: a 1024² image for the 1024-pixel networks, and the native frame for the others.
- Each frame is resized to 1024² (bilinear) and rescaled linearly from its own 1st and 99th intensity percentiles inside
  a central disk. No dataset-wide statistics and no annotation metadata are used.

### 3.2 Dense U-Nets (20 networks)

- **Base ConvNeXt-T U-Nets (5, one per fold).** U-Net with an ImageNet-22k ConvNeXt-T encoder (timm), a 1-channel stem, a
  GroupNorm decoder and two shallow convolution branches at strides 1 and 2 for thin detail. Target: the per-pixel
  annotator mean plus a boundary channel. Loss 0.5 BCE + 0.5 soft Dice + 0.15 boundary BCE. 120 epochs at 1024², batch 8,
  AdamW (learning rate 2e-4, cosine), BF16, flips, rotations and shift-scale-rotate augmentation, EMA of the weights
  (0.999); the EMA checkpoint is chosen on a whole-month slice of the fold's own training months.
- **Small-filament specialists (5).** The base maps miss small filaments. Each base fold model is fine-tuned for 12 epochs
  on native 2048-pixel data: 768-pixel crops, half of them centred on a small annotated filament; BCE weighted by
  1 + 3·m, where m marks annotated instances under 1000 pixels (dilated by 3 pixels), plus soft Dice and boundary terms;
  learning rate 1e-4 (cosine), EMA 0.998. They predict the full 2048² frame.
- **ResNet-34 U-Nets (5).** A second family at 1024² (two-stage recipe: 768 px, then 1024 px with frozen batch norm),
  used only through vote features.
- **Context U-Nets (5).** ConvNeXt-T U-Nets whose three input channels are the image and the out-of-fold maps of the
  ConvNeXt and ResNet-34 families (40 epochs, ImageNet initialisation). Fold k is trained on fold k's out-of-fold maps;
  at test time each fold's context network is applied to that fold's own first-level maps and the outputs are averaged.
  They are used only through vote features.
- All dense networks average 8 flipped and rotated views (D4 test-time augmentation).
- Out: the fused map p = (base map upsampled to 2048² + specialist map) / 2, plus the ResNet-34 and context maps. The
  refiner also needs the five per-fold fused maps, not only their mean.

### 3.3 Candidates

- In: the fused map. Out: candidate instances.
- Connected components of p > 0.4 with at least 150 pixels, holes filled. Components are disjoint by construction.

### 3.4 Per-candidate inputs of the keep model (59 columns)

- **46 map features**: 26 shape, intensity and probability statistics; 14 features on agreement between the plain and the
  D4 view, core fractions, peak, position on the disk, neighbours within 100 px, rank and image foreground; 6 vote
  features from the ResNet-34 map (mean, 10th percentile, fraction ≥ 0.5 and maximum of its probability inside the
  candidate, best IoU with its components, overlap count).
- **Crop CNN (5 networks).** A ResNet-18 (timm) on 128² crops with five channels (image, fused map, plain-view map,
  candidate mask, other candidates) plus the log crop size, trained with soft BCE on u and on the fraction of records
  matched. Cross-fitted on the five folds.
- **Six context votes**: the same six vote features computed on the context networks' map.
- **Six detector columns** from twenty YOLO26m-seg instance detectors (Ultralytics):
  - Training: official COCO weights, one class, input 1536 pixels, 80 epochs, batch 4, AdamW, mosaic, flips on both axes.
    One training sample per annotator record; polygons are the outer contours of the scorer's own rasterisation of each
    instance. Four seeds × five folds, cross-fitted on the same folds.
  - Inference: the native frame, confidence ≥ 0.05, up to 300 instances.
  - Columns per candidate and detector: best mask IoU with a detector instance; that instance's confidence (if IoU ≥ 0.1);
    the highest confidence among instances with IoU > 0.5; intersection over the candidate; intersection over that
    instance; number of instances with confidence ≥ 0.1 overlapping by at least 20% of the smaller area.
  - Each column is averaged over the detectors allowed to score the source: on TRAIN-OOF the four held-out-fold models
    (one per seed), on calibration, assessment and test all twenty (the mean over each seed's five fold models, then
    over the four seeds).
  - The detector masks are used only to compute these columns. Detector instances are never added as masks.

### 3.5 Keep model

- In: the 59 columns. Out: u, the predicted matched IoU averaged over annotator records.
- Bagged gradient-boosted trees (scikit-learn HistGradientBoosting), 20 source-bootstrap bags for each of 5 seeds, 100
  models in all; u is the mean of their predictions, clipped to [0, 1].
- Target: for each TRAIN-OOF candidate, its matched IoU averaged over the source's annotator records (0 for a record it
  does not match).
- Trained on TRAIN-OOF candidates only (cross-fitted over the five folds for TRAIN-OOF scores; a full-TRAIN fit for
  calibration and assessment). The test model is the full refit on 6,733 candidates from 707 sources.
- A candidate is kept if u ≥ 0.25. The threshold was fixed on calibration data and agrees with the PQ/2 rule.

### 3.6 Refiner (5 networks)

- In: a candidate, the image and the fused maps. Out: a refined probability for that candidate.
- Which candidates: a refined probability is computed for every candidate whose FZ verifier score u_FZ is at least
  0.10 (the FZ verifier is the same tree-ensemble recipe on the 46 features and the crop-CNN score only, fitted
  separately); a kept candidate is re-drawn only if u_FZ ≤ 0.55. About 40% of
  kept instances are confident and keep their outline, which roughly halves the refiner's work; this gate is neutral
  for PQ.
- Network: ConvNeXt-T U-Net with stride-1 and stride-2 input branches (32 M parameters).
- Input crop: the candidate's minimum-area rectangle rotated so its long axis is horizontal, padded, at native scale
  (128 × 384 or 192 × 768). Four channels: image, fused map, the candidate's own mask, and the other candidates weighted
  by their score.
- Target: for each annotator record, the annotated instance with the highest IoU with the candidate, accepted if that
  IoU ≥ 0.20; the target is the mean over accepted records.
- Training: 0.5 soft BCE + 0.5 soft Dice, 40 epochs, AdamW, EMA 0.998, epoch chosen on the inner month slice. Our final
  recipe adds 7,126 extra natural seeds from other threshold levels and single-family maps (2.9 times more natural
  crops), and synthetic truncation: in each epoch a training crop is replaced, with probability 0.3, by a copy of its
  seed cut to 40–80% of its length (sometimes eroded), while the target stays the full consensus.
- Cross-fitting and input parity: a TRAIN source is refined only by the fold that held it out, on its out-of-fold fused
  map. On other sources, refiner fold k is fed the fused map of dense fold k (not the 5-fold mean, which is smoother)
  and the five outputs are averaged, each over four flips.

### 3.7 Outline, paste-back and sub-pixel shift

- Re-drawn candidates: the refined probability is moved by (−0.23, +0.05) pixels (rows, columns), thresholded at 0.60;
  only the part connected to the candidate is kept and holes are filled. Contested pixels go to the higher refined
  probability.
- Other kept candidates: the expected-IoU outline (Nowozin, CVPR 2014). Among ten nested level sets of the fused map
  (t = 0.25, 0.30, …, 0.70, each grown only into the candidate's own territory) we take the set S that maximises
  `sum_{i in S} p_i / (|S| + sum_{i not in S} p_i)`. The rule has no tuned parameter. The level is re-cut from the fused
  map moved by the same (−0.23, +0.05) pixels.
- Out: pixel-disjoint masks, one RLE row per filament.

## 4. Results by stage (held-out assessment, 80 sources)

Each delta is against the stage before it, with its 95% month-block interval. Public scores are the leaderboard's
two-decimal display of the corresponding full refit.

- Baseline, one ResNet-34 U-Net with a 26-feature filter: PQ .42311; public 0.36.
- A, five ConvNeXt-T U-Nets and the out-of-fold keep model: .45667, +.0336 [+.0169, +.0546]; public 0.38.
- F, A plus the expected-IoU outline: .45843, +.0018 [−.0005, +.0038]; public 0.39.
- FZ, F plus the small-filament specialists: .47355, +.0151 [+.0063, +.0246]; public 0.39.
  - Recall of annotated filaments under 1000 pixels .532 → .603.
- Round 1, context votes and refiner (first recipe): .47958, +.0060 [+.0007, +.0121]; public 0.40.
  - TRAIN-OOF +.0019 [−.0009, +.0044]; calibration +.0121 [+.0034, +.0242]; positive in every leave-one-month-out subset.
  - Refiner alone +.0049 [+.0016, +.0095]; votes alone +.0015 [−.0026, +.0057].
- Round 2, refiner with truncation training: .47846, −.0011 [−.0026, +.0001]; public 0.40.
  - TRAIN-OOF +.0008 [−.0009, +.0025]; calibration +.0023 [−.0002, +.0044]. Adopted because the rule fixed in advance
    allowed it (pooled with calibration +.0003).
- Round 2, plus detector columns (one detector seed): .48142, +.0030 [−.0040, +.0135].
  - TRAIN-OOF +.0027 [+.0005, +.0048]; calibration +.0069 [+.0027, +.0106] (measured with the first refiner).
- Round 2, plus the sub-pixel shift: .48225, +.0008 [−.0009, +.0030]; public 0.40 (an intermediate upload).
- Round 3, detector columns averaged over four seeds (final): .48065, −.0016 [−.0059, +.0012] against one seed.
  - Adopted under a non-inferiority rule fixed in advance (margin −.002) as the lower-variance version of the same
    model: the single seed used in round 2 was the luckiest of four on calibration, and averaging four seeds lowers the
    keep model's held-out loss on TRAIN-OOF by 4.6% (one seed: 2.5%).
- Final against the previous submission (round 1): TRAIN-OOF +.0038 [+.0009, +.0068]; calibration +.0106
  [+.0031, +.0152]; assessment +.0011 [−.0062, +.0113]; pooled calibration and assessment +.0051.
- How to read this: the out-of-fold keep model and the specialists are the large steps. After round 1 the steps are a
  few thousandths each; they are positive on TRAIN-OOF and calibration, but on 80 assessment sources none of their
  intervals excludes zero, and the public board shows 0.40 for every pipeline since round 1.
- Of the 19 arms of the round-2 run, the arm with the first refiner, detector columns and the shift scored highest on
  assessment (.48347). It was not selectable after the fact, and we did not select it.

Overlap statistics of the final pipeline on assessment (pooled over the 140 annotator records):
- 666 matched filaments, 241 false positives, 342 misses; SQ 0.691, RQ 0.696.
- Matched IoU: median 0.695 (10th–90th percentile 0.574–0.798). Matched Dice: median 0.820.
- 27 of 1008 annotated filaments are split across two or more predictions; 13 of 907 predictions cover two filaments.
- Recall by annotated area: 0.48 under 500 pixels, 0.73 at 4000–8000 pixels.

## 5. The sub-pixel offset

- Measurement: for every matched (prediction, annotation) pair we compute IoU with the prediction moved by one pixel up,
  down, left and right. The vertex of the parabola through the three values on each axis is the IoU-optimal shift.
- TRAIN-OOF, 3,820 pairs: −0.234 pixel in rows [−0.260, −0.210] (towards the first row of the image) and +0.049 pixel in
  columns [+0.027, +0.070]. Mean matched IoU 0.6872 unshifted, 0.6787 one pixel up, 0.6637 one pixel down.
- Independent checks:
  - calibration, 476 pairs: −0.270 [−0.336, −0.191] in rows, +0.020 [−0.043, +0.076] in columns;
  - a Mask2Former we trained on the same labels for another experiment: −0.216 [−0.243, −0.187] in rows, +0.046 in
    columns;
  - the public code of the team BarrelAged moves selected masks 0.23 pixel towards row 0.
- Likely cause (not tested directly): our pipeline is symmetric under flips, through training augmentation and test-time
  averaging, while the scorer's rasterised drawings are not. A model averaged over flipped views cannot express a
  one-directional sub-pixel offset.
- Correction: fixed at (−0.23, +0.05) from the measurement, not tuned on PQ. About 2% of mask pixels change and no mask
  disappears. The remaining offset after the shift is −0.058 / +0.011 pixel.
- Effect: TRAIN-OOF +.0006 [−.0000, +.0013]; calibration +.0015 [+.0003, +.0027]; assessment +.0008 [−.0009, +.0030]
  inside the pipeline. An earlier check of ours had tested a different shift and missed the offset.

## 6. What did not work

Each item was judged by a rule written before its data existed.
- **Mask2Former** (Swin-S, one per fold, two seeds, settings following the public recipe of the team supersede).
  - On its own it matches our whole dense pipeline on TRAIN-OOF: 0.4574 and 0.4592 against 0.4596.
  - As a voter its gain is seed-dependent on calibration: TRAIN-OOF / calibration +.0044 / +.0021 (seed 0) and
    +.0028 / −.0012 (seed 1).
  - Its extra instances are almost never drawn (mean target .01–.10).
  - Its outlines are no better than ours: matched IoU 0.6904 against 0.6907 on 3,684 common matches.
- **Mask R-CNN** (torchvision R50-FPN v2, two seeds): 0.4224 / 0.4327 on its own; as a voter +.0030 / −.0001.
- **A stronger detector** (YOLO26x-seg at 2048 pixels): better on its own (0.4342 against 0.4171) but weaker as a voter
  (+.0032 / +.0002).
- **Combinations of voter families.** Every family added on top of YOLO gains about .001–.002 on TRAIN-OOF and loses
  about .002–.005 on calibration. The only subset that passed its rule ({YOLO, Mask2Former, YOLO26x}) lost on assessment:
  −.0004 [−.0065, +.0073] against the four-seed keep.
- **Detector-only instances** as extra masks: mean target .16–.18 against the .25 bar. Eroding detector masks by one
  pixel lowered matched IoU (0.6667 → 0.6607).
- **Applying the keep model fold by fold** (scoring with one fold's detector columns at a time and averaging the
  scores, to match how the model is fitted): −.0030 [−.0079, +.0000] on calibration.
- **An image-only appearance scorer** (ConvNeXt-T on the refiner crops, image and seed only) as one more column: passed
  its rule narrowly (+.0006 TRAIN-OOF, +.0014 calibration per bag seed), lost on assessment (−.0010). With the
  probability map in its input it was a near-copy of the keep model (correlation 0.93) and changed nothing.
- **Refiner variants**: probability-channel dropout (−.004 to −.005 on calibration), hard-seed weighting (−.005),
  two-pass refinement (negative), weight averaging of two refiners (−.0105 on calibration). No refiner recovers
  filaments that are cut short where the dense maps are blank: about half of the missing area of truncated candidates
  has fused probability below 0.1, and refiners cover about 1% of it or less.
- **Earlier rounds**: larger encoders, self-supervised pretraining on unlabelled frames, label transfer to nearby frames,
  pseudo-labels, skeleton-recall, Dice-semimetric and blob losses, copy-paste, seed averaging of the dense networks,
  fragment merging (a pre-registered rule gave −.0004 [−.0037, +.0020]), selection from a multi-level candidate pool, and
  53 outline policies did not beat seed noise or the simpler choice.

## 7. Limitations

- **Annotator disagreement caps the score.** On multi-annotator TRAIN images another annotator matches 62.9% of a given
  annotator's filaments; our pipeline matches 64.4%. Where two annotators agree we match 81.9%. 40% of our false
  positives match a different annotator of the same image. An oracle that picks the best disjoint subset of the
  annotators' own masks reaches 0.773 (0.644 on multi-annotator images). Replacing our masks by one expert's drawing and
  scoring against another expert lowers PQ from 0.4585 to 0.3680.
- **Evaluation.** The 80 assessment sources have served seven pre-registered single runs. Their 14 months make intervals
  wide; the gains after round 1 are below what they can resolve.
- **Cost.** 50 networks per frame (20 dense U-Nets, 5 crop CNNs, 5 refiners, 20 detectors) and two tree ensembles
  (the FZ verifier and the keep model). On one RTX 5090 with 32 vCPUs the whole pipeline takes a median of 24.7 s per
  frame (76 minutes for the 180 test frames); 13 s of it is the detector stage, mostly the CPU-side computation of
  the agreement columns. FZ alone takes 238 s per frame on 8 CPU cores. The README has the breakdown.
- **Licence.** The detector library and its starting weights are AGPL-3.0 (section 9).
- **Data.** No external labels. Unlabelled GONG frames (test times ±1 hour excluded) were used only in experiments that
  did not work.

## 8. Prior work in this competition

We read these sources as text; we did not run their code or use their weights or outputs.
- tushhaaarrrr, public Kaggle notebook (Sept 30 2026),
  <https://www.kaggle.com/code/tushhaaarrrr/solar-filament-segmentation-2026-0-40>: cross-fitted scoring of candidates by
  expected matched quality; the derivation that one global threshold of PQ/2 is optimal; an instance-conditioned crop
  refiner (rotated native-scale crops with image, probability, seed and neighbour channels; consensus target;
  seed-connected paste-back). Our keep model and refiner implement these ideas in our own code.
- BarrelAged, <https://github.com/Barrel-Aged/filament-segmentation-2026>: 17 agreement features per detector
  (including Mask R-CNN) as inputs of a candidate scorer, and a shift of selected masks by 0.23 pixel towards row 0
  (`filseg/shift.py`). Our contribution here is a cross-fitted, pre-registered test of detector agreement inside our
  pipeline and an independent measurement of the offset.
- supersede (cosmicwitness), <https://github.com/Nuwdawg/Sol>: native-resolution Mask2Former recipe (we followed its
  settings for our Mask2Former experiment) and YOLO detections as votes.
- notoverkil, <https://www.kaggle.com/datasets/notoverkil/solar-filament-weights>: a five-family detector and U-Net
  ensemble with a gradient-boosted rescorer and crop refiners.

## 9. Licences of the components

As declared in the package metadata of the versions pinned in `requirements.txt` (the README's "Third-party
components" lists the same):
- Ultralytics (YOLO26m-seg training and inference, `ultralytics-opencv-headless` 8.4.210, with ultralytics-thop) and
  the Ultralytics COCO weights we started from: AGPL-3.0. The repository's own licence is stated in the README.
- PyTorch 2.11.0 and torchvision 0.26.0: BSD-3-Clause. timm 1.0.31 (ConvNeXt-T and ResNet encoders): Apache-2.0; the
  ImageNet weights we started from keep the licences of their original releases (ConvNeXt: MIT; timm's ResNet-18:
  Apache-2.0; torchvision's ResNet-34: BSD-3-Clause).
- scikit-learn 1.9.1 (FZ verifier and keep model), NumPy 2.5.4, SciPy 1.18.1: BSD-3-Clause. pycocotools 2.0.11:
  BSD-2-Clause. OpenCV (headless) 5.0.0.93 and safetensors 0.8.0: Apache-2.0.
- The competition data (MAGFiLO, GONG H-alpha) is not included and keeps its own terms.

## 10. References

- Ahmadzadeh et al., A dataset of manually annotated filaments from H-alpha observations, Scientific Data 11, 1031
  (2024), doi:10.1038/s41597-024-03876-y.
- Kirillov et al., Panoptic Segmentation, CVPR 2019.
- Ronneberger et al., U-Net, MICCAI 2015. Liu et al., A ConvNet for the 2020s (ConvNeXt), CVPR 2022. He et al., Deep
  Residual Learning, CVPR 2016. He et al., Mask R-CNN, ICCV 2017.
- Nowozin, Optimal Decisions from Probabilistic Models: the Intersection-over-Union Case, CVPR 2014.
- Wolpert, Stacked Generalization, Neural Networks 5(2), 1992.
- Künsch, The Jackknife and the Bootstrap for General Stationary Observations, Annals of Statistics 17(3), 1989.
- Pedregosa et al., Scikit-learn, JMLR 12, 2011.
