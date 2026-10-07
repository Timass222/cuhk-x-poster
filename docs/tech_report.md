# CUHK-X Small Model Track — Technical Report (team Timass)

Draft of 9 Sep 2026. Numbers marked *(pending)* are filled from the Stage-2 rehearsals of 9 Sep.
Companion documents: `README.md` (how to run, rehearsal logs), `docs/findings.md`
(the complete experiment log, Russian), `docs/rules_qa.md` (rule interpretations).

## 1. Summary

* Task: 40-class activity recognition on the CUHK-X multimodal clips (Depth_Color, IR, Thermal,
  Skeleton, IMU, Radar), cross-subject test (users unseen in training), one model file ≤ 100 MB.
* Training data: 3 036 clips, 18 users (133–203 clips each), 40 classes. Test: 405 clips from
  other users, recorded in 16 sessions (recovered from frame timestamps).
* Final system (`checkpoints/model.pth`, 93.99 MiB): a late fusion of five branches — VideoMAEv2
  ViT-B/16 on 224 px person crops of colourised depth + IR (int6), VideoMAEv2 ViT-S/16 on IR crops
  (int5), CTR-GCN on 2-D skeletons (3 folds, int8), gradient boosting on physical IMU features, and a
  small IMU network (5 folds, int8) — followed by a deterministic, label-free session decoder
  (repeat pooling, second-order transitions, MAP repair), session-level prototype adaptation in
  the ViT feature space and an EM re-estimation of the transition tables on the sample itself.
* Public leaderboard 0.86069 (173/201 clips), rank 11 on 8 Sep. The packaged pipeline reproduces
  the submitted CSV from the raw Kaggle archive on 405/405 clips.
* The single largest finding of the last week: the composition of the training set matters more
  than any modelling choice we tried. Excluding one training user (user9) from the ViT-B training
  set moved the public score from 0.831–0.836 to 0.861 (+5 clips), reproducibly across three
  independent trainings; all other levers (quantisation scheme, seeds, weight soups, QAT,
  self-training on pseudo-labels, test-time augmentation, larger backbones) stayed within ±1 clip.

## 2. Data handling and preprocessing (all inside `inference.sh`)

1. **Clip index and sessions.** Clips are ordered by the timestamp of their first frame
   (Depth_Color/IR file names; fallbacks: Radar file name, Skeleton json names, the `time` column of
   the IMU csv). A new session starts when the gap exceeds 1 800 s. Clips without any timestamp
   form singleton sessions (the decoder then degrades gracefully to per-clip fusion).
2. **Depth colour map inversion.** Depth_Color frames are a rainbow palette; a lookup table
   maps RGB back to scalar depth (`code/data/preprocess/colormap.py`), then 128 px tensors are cached.
3. **Person boxes.** YOLO11n (weights inside the pack, fp16) on 12 raw IR frames per clip;
   `n_det ≥ 2 & conf ≥ .40 → yolo`, `n_det ≥ 1 & conf ≥ .25 → yolo_soft`, else a 1 280 px pass, else the
   motion box (per-pixel std of depth over time). Square window, margin 1.40, min side 0.35.
   On the Kaggle test: 382 / 13 / 6 / 4 clips per source.
4. **Crops.** Depth_Color (as RGB) and IR are cropped with the same box at 224 px, 16 frames sampled
   uniformly over the clip, 4 input channels (RGB depth + IR) for the ViT-B, IR only for the ViT-S.
   Only the caches consumed by the branches present in the pack are built.
5. **IMU.** The two csv files (`up`: LA+RA+C, `down`: LL+RL) are parsed into 5 devices × 9 channels
   × 32 steps (resampled), plus 477 physical statistics (linear acceleration, tilt, gyro ×
   10 statistics, sin/cos yaw) for the boosting branch. Device-wise normalisation constants are
   stored in the pack.
6. **Skeleton.** The COCO-17 2-D poses are converted to T × 17 × 3 sequences (x, y, confidence).
7. Thermal and Radar are read only for timestamps; no branch of the final pack consumes them.

## 3. Branches

| branch | input | architecture | init | training | quant. | size |
|---|---|---|---|---|---|---|
| `vit` | depth-RGB + IR crops 224², 16 frames | VideoMAEv2 ViT-B/16 (86.6 M) | `vit_b_k710_dl_from_giant` (distilled from ViT-g, Kinetics-710) | 20 ep., AdamW lr 2e-4, wd .05, LLRD .75, drop-path .2, EMA, bs 16 + grad-ckpt; **17 users (user9 held out)** | int6 per-output-row symmetric | 62.7 MiB |
| `vit_ir` | IR crops 224², 16 frames | VideoMAEv2 ViT-S/16 (22 M) | `vit_s_k710_dl_from_giant` | same recipe, drop-path .1; fold 0 | int5 | 13.2 MiB |
| `ctrgcn` ×3 | skeleton T×17×3 | CTR-GCN | PYSKL NTU-60 (2-D, joint) | 5-fold, folds 0/3/4 shipped | int8 | 3 × 1.7 MiB |
| `imu_hgb` | 477 IMU statistics | HistGradientBoosting (150 iters, 15 leaves) | — | all 18 users | pickle, gzip | 4.0 MiB |
| `imu_v2` ×5 | 5 × 9 × 32 IMU tensor | small conv net (stems 32/64, temporal 160, head 128) | — | 5-fold, all folds shipped | int8 | 5 × 0.44 MiB |

Other pack contents: YOLO11n fp16 5.35 MiB, decoder tables + adaptation prototypes 0.58 MiB.
Out-of-fold (5-fold, users grouped) accuracies of the single branches on the training set:
ViT-B .761 (balanced .73), ViT-S IR balanced ≈ .68, CTR-GCN .567 (balanced .511), IMU HGB ≈ .39,
IMU net .396 (balanced .378). The full stack after the session decoder reaches .846 / .862 on the
two halves of the cross-fold harness (see §5); the decoder itself is worth ≈ +8–11 pp over
per-clip fusion.

## 4. Fusion and session decoding

* **Fusion**: temperature-scaled arithmetic mean of branch posteriors with fixed weights
  (ViT-B .40, HGB .21, ViT-S IR .20, CTR-GCN .075, IMU net .06; HGB temperature 1.58, CTR-GCN 1.08),
  τ = .85 sharpening, class prior of the training set. Weights were set by a coordinate scan on the
  out-of-fold predictions with a cross-fold acceptance rule (must not hurt on either half of the
  folds).
* **Session decoder** (deterministic, no labels): within a session, consecutive clips of the same
  action are pooled (repeat pooling); a second-order transition model learned on the training
  sessions (action order is not random in the protocol) is applied; labels inside a block are
  made distinct (MAP repair with a duration prior). Gain on the harness ≈ +8–10 pp accuracy over
  per-clip fusion; on the Kaggle test the decoder changes 67–73 of 405 answers.
* **Prototype adaptation**: after the first decoding pass, class prototypes are built from the ViT-B
  features of the session's own clips (decoded labels, κ = 3, threshold .05) and mixed into the
  ViT posterior with weight .3; a second decoding pass follows. Changes ≈ 30 pre-decoder and
  ≈ 10 final answers on the test.
* **EM re-estimation** of the transition tables on the sample's own session structure (λ = .3, one
  iteration): changes 1–4 final answers.
* **Fallback without sessions**: if no timestamps can be recovered, every clip is its own session;
  the pipeline still runs and returns per-clip fusion (measured: 73/405 answers differ from the
  session decode on the Kaggle test, i.e. the decoder is worth ≈ 8–10 pp there).

## 5. Validation protocol and the training-set finding

* 5 folds grouped by user (fold 3 = users 9, 18, 24). Every design decision was gated on
  out-of-fold predictions with a two-sided cross-fold harness (a change must not hurt on either
  half), never on the public leaderboard.
* The public leaderboard was used only to check the transfer of validated changes; each probe was
  a complete packaged pipeline.
* **Composition finding.** ViT-B models trained with all 18 users, or with any fold other than
  fold 3 held out, score 0.831–0.836 on the public set; models trained without users 9/18/24
  (fold 3) score 0.851–0.861 over three seeds; removing user9 alone gives 0.861, removing only
  user18 or user24 gives 0.851. Out-of-fold analysis shows why: on the object-in-hand classes
  (Read documents, Eat food, Take medicine, Drink water) these three users execute the actions
  atypically (recall by models trained on the other users .22–.39 vs .57–.67), and models that
  learned from their clips shift test predictions into those classes. The shipped ViT-B is
  therefore trained on the 17 other users with user9 as an honest held-out validation set
  (accuracy .818 on user9, balanced .816). No test labels were involved in this analysis.
* What did **not** help (all within ±1 clip on the public set, details in `findings.md`):
  quantisation schemes (int7, group-wise, MSE-clipped, QAT), weight soups, probability averaging of
  seeds, test-time augmentation (6 views), mixup, 32-frame input, larger backbone (ViT-L does not
  fit the size cap), self-training on consensus pseudo-labels (permitted by the rules; 4 answers
  changed, −1 clip), hand-region crops (an OOF arbiter gate rejected them).

## 6. Packaging

One `.npz` (`checkpoints/model.pth`, 93.99 MiB / 98.55 MB decimal): every weight tensor is stored
as a row-wise symmetric integer code (int6 for the ViT-B, int5 for the ViT-S, int8 elsewhere) with
fp32 per-row scales; the boosting model as a gzipped pickle; the YOLO detector in fp16; plus the
fusion config, adaptation prototypes (40 × 768 fp16), decoder tables and the branch
specification (`branch_spec`, `branch_mods`) that lets `inference.sh` build only the caches the
pack needs. Quantisation cost on the held-out fold: fp32 .7896 → int6 .7761 accuracy for the
ViT-B alone; after fusion and decoding the packaged output differs from the fp32 research decode
on ≤ 1 clip of 405.

## 7. Reproducibility and efficiency

* `inference.sh <sample.zip|dir> <out.csv>` runs the whole pipeline from the raw archive; the
  packaged output reproduced the submitted CSV on 405/405 clips in every rehearsal (README §4).
* The pipeline is deterministic (stable sort of clips by `(first_ts, clip)`, fixed seeds, no
  randomised augmentation at inference).
* Robustness checks of 9 Sep (all with the final pack): (a) a 60-clip sample given as a
  **directory** with a different top-level folder name, no Thermal/Radar folders anywhere, 9 clips
  without IMU, 6 without skeleton, 9 with frame names carrying no timestamp, 2 clips with 3 frames
  and 1 empty clip folder — runs in 183 s, every clip gets a row, 4 of 59 answers differ from the
  full-set decode (one empty clip, one 3-frame clip, one skeleton-less clip, one context effect);
  missing IMU or missing timestamps changed nothing. (b) A 5-clip archive: 88 s, 1 of 5 answers
  differs (shorter session context). (c) These runs found and fixed five defects before the
  freeze: CPU-only device selection on Windows, missing-IMU handling for directory input, empty clip
  folders dropped from the index, a hard-coded `Thermal_files` column, and the output path of a
  file-less clip.
* Timing on the Kaggle test (405 clips, 16 sessions), RTX 3060 Ti 8 GB, GPU exclusive, with the
  pack-aware cache building of 9 Sep (README §4d): **552 s = 9 m 12 s** end-to-end from the raw
  zip (meta 1 s, depth frames 76 s, motion boxes 3 s, YOLO 93 s, 224 px crops 195 s, IMU + skeleton
  11 s, branch inference + two decoder passes 170 s) → ≈ 1.36 s per clip, ≈ 45 min per 2 000 clips.
  Earlier runs of the same pack: 645 s (8 Sep, cold caches, all caches built), 1 034 s with concurrent
  CPU load. Peak GPU memory 2.9 GB on the device (≈ 1.8 GB above the idle desktop baseline); the
  packaged models fit a 4 GB GPU. CPU-only run (`CUDA_VISIBLE_DEVICES=`, same 405 clips): **1 219 s = 20 m 19 s**, output identical to the
  GPU run on 405/405 clips (YOLO 179 s, crops 190 s, branch inference + decoding 741 s); i.e. the
  whole pipeline also fits the 2-hour budget without a GPU (≈ 100 min per 2 000 clips).
* Model size 93.99 MiB; parameters ≈ 86.6 M (ViT-B) + 22 M (ViT-S) + 3 × 1.5 M (CTR-GCN) + 5 × 0.1 M
  (IMU net) + HGB.

## 8. External data and pretrained weights (disclosure)

* VideoMAEv2 ViT-B/16 and ViT-S/16 distilled from ViT-g on Kinetics-710
  (OpenGVLab/VideoMAEv2, `distill/vit_b_k710_dl_from_giant.pth`, `distill/vit_s_k710_dl_from_giant.pth`;
  Apache-2.0 / MIT code). Fine-tuned by us on the CUHK-X training split only.
* CTR-GCN initialised from a PYSKL NTU-60 (2-D, joint) checkpoint (Apache-2.0).
* YOLO11n person detector (Ultralytics, AGPL-3.0), used only to cut crops.
* Earlier packs (not in the final model) used R(2+1)D-34 IG65M→Kinetics-400 (facebookresearch/VMZ)
  and torchvision R(2+1)D-18 Kinetics-400; UMT-B and InternVideo2 were screened and not used.
* No external CUHK-X data, no test labels, no manual labelling of test clips. Pseudo-labelling of
  the test set was tried once (permitted by the organisers' Q&A) and not used in the final model.

## 9. Code map

* `inference.sh`, `code/infer/{make_test_meta,make_boxes,pack_infer,assemble_pack,pack_checkpoint}.py`
* `code/data/preprocess/{frames,colormap,bbox,crop_cache,imu,skeleton,build_caches}.py`
* `code/train/{train_video,train_skeleton,train_imu_hgb_v2,train_imu_v2,cv}.py`,
  `code/models/branches/*`, `code/models/fusion.py`
* `code/scratchpad_snapshot/` — research scripts referenced in `docs/findings.md`.
