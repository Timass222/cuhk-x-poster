# Team Timass Reproduction Guide

This is the guide that shipped with the frozen verification package. In this public repository,
`artifacts/model.pth` is not stored in Git: download it from the GitHub release (see README.md).
Some files named below exist only in the private package and are not published here: `docs/findings.md`,
`docs/rules_qa.md`, `declarations/`, `archives/`, `logs/` and `artifacts/model_pack3ira.pth`.

Verification package for the CUHK-X Challenge, Small Model Track.

## 0 Needs attention (answer first)

- More than one GPU needed: `no`
- More than 24 GB of GPU memory needed: `no` (peak: `2.9 GB` total on device, ≈1.8 GB above idle baseline; measured on an RTX 3060 Ti 8 GB)
- Specific GPU generation or driver needed: `no` (CUDA 12.6-capable driver recommended for the pinned cu126 wheels; the pipeline also runs CPU-only)
- Network access needed at inference time: `no` (verified with Docker `--network none`; network is only needed once, to install pinned Python packages)
- Runs on Linux x86_64: `yes` (developed and rehearsed on Windows 10 + Git Bash; the Dockerfile targets `python:3.13.15-slim-bookworm` on Linux x86_64)

## 1 Team and submission

- Team name: `Timass`
- Track: `small`
- Selected Kaggle Final Submission record: `submission/submission_info.yaml`
- Selected final submission(s):
  - #1 (primary): Kaggle submission ID `56094379`, file name `submission_pack4_17u9_infer.csv`, submitted `2026-09-08T08:25:14Z`, private score `0.86764`, SHA-256 `970d4753cde6372d5ca343c13ed77e8cb20fec300d860e12e676829fe84f91aa`
  - #2: Kaggle submission ID `56058031`, file name `submission_pack3ira_w20_infer.csv`, submitted `2026-09-06T14:59:37Z`, private score `0.86274`, SHA-256 `90d9ed7eb694bb14817968627aaea2cca17a45205538dc332739940df6e54268`
- Primary verification target: `#1` (submission ID `56094379`, higher private score)
- Solution type: `local_checkpoint`

Both Selected submissions share the identical public leaderboard score (0.86069); their private scores differ, and #1 (`56094379`, pack `PACK4_17U9`) is higher and is therefore `primary_verification_submission_id`.

## 2 Frozen solution summary

Late-fusion ensemble over five unimodal branches, all trained on the CUHK-X Small Model Track training split only (no external labelled data):

- `vit` — VideoMAEv2 ViT-B/16 (86.6 M params), depth-RGB + IR crops, 224 px, 16 frames. Initialised from a
  publicly distilled Kinetics-710 checkpoint, fine-tuned by us. Quantised int6 for the pack.
- `vit_ir` — VideoMAEv2 ViT-S/16 (22 M params), IR crops, 224 px, 16 frames, same pretraining lineage,
  fine-tuned fold 0. Quantised int5.
- Skeleton — CTR-GCN, folds 0/3/4, initialised from a PYSKL NTU-60 2-D joint checkpoint. Quantised int8.
- IMU (statistical) — HistGradientBoosting (scikit-learn) on 450 hand-built features.
- IMU (network) — a small IMU network, 5 folds, int8 (pack #1) or an IMU TCN x3 (pack #2, no int quantisation
  difference in kind, same slot).
- Person detector — YOLO11n (Ultralytics), fp16, used frozen only to crop a person-centred window from raw
  IR frames; not fine-tuned.

Preprocessing (deterministic, shared between training and inference): depth colour-map inversion to scalar
depth, motion-box detection from Depth_Color (per-pixel std over time), YOLO11n person boxes on 12 raw IR
frames per clip with a fallback cascade (`yolo` → `yolo_soft` → 1280 px pass → motion box), then 224 px
hybrid crops (Depth_Color as RGB + IR) for the ViT branches and IMU/skeleton tensor building.

Inference-time fusion and decoding: temperature-scaled arithmetic fusion across branches (temperature fit on
out-of-fold predictions, never on training folds); a session decoder pass (repeat pooling, second-order
transition tables, distinct-label-per-block repair) using clip timestamps to group clips into sessions;
session-level prototype adaptation in the ViT feature space (cosine similarity to the decoded labels of the
same session, weight 0.3); a label-free EM re-estimation of the transition tables on the sample's own session
structure (λ 0.3, 1 iteration, pack #1 only); a second decoder pass. All of this uses no labels and is
deterministic given the input.

**What differs between the two Selected Submissions** (both packs run through the same `inference.sh` /
`src/infer/pack_infer.py` path, selected by the third argument / `submission_id`):

| | #1 primary (`artifacts/model.pth`, submission 56094379) | #2 (`artifacts/model_pack3ira.pth`, submission 56058031) |
|---|---|---|
| `vit` training split | 17 of 18 training users (user9 held out) | fold-3 split (users 1-8, 16, 17, 19-23 in training) |
| IMU network slot | small IMU network, 5 folds, int8 | IMU TCN, 3 folds |
| EM re-estimation | yes (λ 0.3, 1 iteration) | no |
| Everything else (ViT-S/IR, CTR-GCN, HGB, YOLO11n, decoder, adaptation) | identical | identical |

Model size: 93.99 MiB per pack (≈86.6 M + 22 M + 3×1.5 M + 5×0.1 M parameters, quantised).

No LMT (Large Model Track) data was used in either Selected pack. Pre-/co-training with the organizer-released
HAU/HARn data was explored (permitted per organizer Q&A) but every HAU-augmented variant scored below the
Selected packs on the public leaderboard and was not used; see `docs/findings.md` (15 Sep entries) for the
full comparison. No manual test labelling, no test ground truth, no external CUHK-X data.

## 3 Hardware and runtime

- Operating system: Linux x86_64 (Dockerfile: `python:3.13.15-slim-bookworm`); developed and rehearsed on
  Windows 10 + Git Bash, both give byte-identical output
- GPU model or minimum compute capability: any CUDA-capable NVIDIA GPU (developed/tested on RTX 3060 Ti 8 GB); CPU-only also works
- GPU count: 1
- Minimum GPU memory per GPU: 4 GB (peak measured 2.9 GB total on device)
- Minimum host memory: not separately profiled; comfortably fits in 16 GB (two ViT backbones + frame/crop
  caches for a few hundred clips in memory at once)
- Free disk space required: ~15 GB (package ≈200 MB; per-run working caches under `work/` scale with the
  number of clips in `<data_dir>` — frames/crops/features, cleared/rebuilt per run, not part of the package)
- Expected environment setup time: 10-20 min (pip install of the pinned cu126 wheels; no other external downloads)
- Expected inference time: ~9-10 min for 405 clips on GPU (RTX 3060 Ti, exclusive); ~20-25 min CPU-only.
  Throughput budget documented in the development repo: ≈45 min per 2000 clips on GPU.

## 4 Environment setup

**Docker (preferred):** `environment/Dockerfile` builds a self-contained Linux image with the exact pinned
dependencies and both packs. Build context is the **package root** (the Dockerfile references `code/`,
`artifacts/`, `environment/` relative to it):

```bash
docker build -f environment/Dockerfile -t timass-cuhkx .
docker run --rm --gpus all --network none --shm-size 16g \
    -v <data_dir>:/data:ro -v <output_dir>:/output \
    timass-cuhkx  bash inference.sh /data /output/submission.csv
```

This package does not include a pre-built `docker save` archive or a registry tag; build locally from
`environment/Dockerfile` with the command above. Without `--gpus` the same image runs on CPU. The base image
already installs the two system packages OpenCV/ultralytics need at import time (`libgl1`, `libglib2.0-0`).

**Without Docker**, on a clean Linux (or Windows + Git Bash) machine with Python 3.13.15:

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows Git Bash: source .venv/Scripts/activate
pip install --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.org/simple \
    -r environment/requirements.txt -c environment/constraints-venv.txt
```

Exact pinned versions (`environment/requirements.txt`): torch 2.13.0+cu126, torchvision 0.28.0+cu126,
numpy 2.5.2, pandas 3.0.5, scipy 1.18.1, **scikit-learn 1.9.0** (the IMU HistGradientBoosting model is a
scikit-learn pickle — keep this version exact), pillow 12.3.0, opencv-python 5.0.0.93, ultralytics 8.4.136
(bundles the YOLO11n architecture code; weights are inside the packs, not downloaded).
`environment/constraints-venv.txt` pins the full transitive dependency set that produced the leaderboard
submissions. No GPU driver beyond CUDA 12.6 support is required; without a GPU, `inference.sh` runs on CPU
automatically (falls back to `python3` if no `.venv` is found next to it, otherwise uses the venv's Python).

Network access is required only for this one-time `pip install` (or `docker build`) step, to reach
`pypi.org` and `download.pytorch.org`. No network access is used while `inference.sh` runs.

## 5 Run inference

From the package root:

```bash
bash inference.sh <data_dir> <output_csv>
# to reproduce the second Selected submission instead of the primary one:
bash inference.sh <data_dir> <output_csv> 56058031
```

With two arguments the script reproduces the primary verification submission (ID `56094379`,
`artifacts/model.pth`). With `56058031` as the third argument it reproduces the other Selected submission
(`artifacts/model_pack3ira.pth`) instead. `primary` is also accepted as an explicit alias for the
two-argument behaviour.

`<data_dir>` must contain the official label-free Testing data in the published Kaggle layout, any top-level
folder name (accepts either an extracted directory or the Kaggle test `.zip` directly):

```
<prefix>/<clip_id>/Depth_Color/Depth_<date>_<time>_<idx>_Color.png
<prefix>/<clip_id>/IR/IR_<date>_<time>_<idx>.png
<prefix>/<clip_id>/Thermal/frame_<idx>.jpg
<prefix>/<clip_id>/Skeleton/predictions/Color_<date>_<time>_<idx>.json
<prefix>/<clip_id>/IMU/up(LA+RA+C).csv , down(LL+RL).csv
<prefix>/<clip_id>/Radar/radar_output_T<date>.csv
```

Missing modalities/clips are tolerated: a branch without input for a clip is skipped by the fusion (its
weight is redistributed among the remaining branches); a clip with no frames at all still gets a prediction
from the IMU/skeleton branches or, failing that, the session decoder.

Output: `path,prediction` written to `<output_csv>` — `path` is the archive-relative clip directory
(`<prefix>/<clip_id>/`, matching Kaggle's `test.csv`), `prediction` is the action id 0-39, in the same clip
order as the input.

The script logs every step (clip indexing, frame/crop caches actually needed by the selected pack, YOLO
detection source counts, branch inference, fusion, decoding) to `work/stage2/inference.log`
(override the work directory with `WORK=<path>`); the same log is echoed to stdout. It exits non-zero with a
message on stderr if `<data_dir>` does not exist or an unsupported `submission_id` is given. All intermediate
caches are written under `work/`, never inside the package's own `src/`/`artifacts/`.

## 6 Model artifacts and external resources

See `manifest.yaml` → `artifacts` for the complete machine-readable list. Summary:

- `artifacts/model.pth` (93.99 MiB, NumPy `.npz`) — primary pack, required, reproduces submission `56094379`.
- `artifacts/model_pack3ira.pth` (93.99 MiB) — secondary pack, required, reproduces submission `56058031`.
- VideoMAEv2 ViT-B/16 and ViT-S/16, distilled from ViT-g on Kinetics-710 — public pretrained backbones,
  fine-tuned by us on the CUHK-X training split only. Source: `github.com/OpenGVLab/VideoMAEv2`, checkpoints
  `distill/vit_b_k710_dl_from_giant.pth` and `distill/vit_s_k710_dl_from_giant.pth` (Apache-2.0 / MIT code).
- CTR-GCN initialised from a PYSKL NTU-60 (2-D, joint) checkpoint (Apache-2.0), fine-tuned by us.
- YOLO11n person detector (Ultralytics, AGPL-3.0), used frozen, not fine-tuned; weights are bundled inside
  both packs.
- No API models or services are used. No credentials of any kind are required or included.

## 7 Reproduction expectations

Inference is deterministic: all branch weights are quantised integers packed into the `.npz` file, and the
fusion/decoding steps use no randomness. Rehearsals on 8-9 Sep 2026 reproduced both
`submission_pack4_17u9_infer.csv` and `submission_pack3ira_w20_infer.csv` on **405/405 clips** from the raw
Kaggle test zip, run from multiple working directories, on both GPU and CPU-only. One rehearsal run where the
GPU was shared with another job produced a single-clip (1/405) difference caused by YOLO detection jitter on
a borderline clip (motion-box fallback selected instead of a soft YOLO box); this did not change the decoded
label and is the only nondeterminism observed. No other precision mode or seed sensitivity is expected —
random seed 0 was used at training time only, not at inference time.

Compare the organizer-run CSV to `submission/final_submission_1.csv` (primary) or
`submission/final_submission_2.csv` row-for-row on `prediction`; expect an exact match, with the single
documented jitter case as the only known source of a possible 1-clip discrepancy.

## 8 Known limitations and troubleshooting

- The pipeline auto-detects whether the selected pack needs IR/Thermal frame caches and Depth/IR crop
  variants, and only builds what that pack actually consumes; the two Selected packs both need the
  224 px hybrid Depth_Color(RGB)+IR crops only, so the 128 px IR/Thermal caches are skipped for them.
- If `<data_dir>` is a directory without a Radar/Thermal subfolder, or without IMU/skeleton files for some
  clips, or with clips whose frame filenames carry no timestamp: these were exercised in a 60-clip
  robustness rehearsal (9 Sep 2026) and are handled (a clip without a timestamp becomes a singleton session;
  a clip missing a modality is decoded from its remaining modalities or the session decoder).
- CPU-only inference (`CUDA_VISIBLE_DEVICES=` before the command, or a machine without a GPU) works within
  the time budget above but is roughly 2-2.5x slower than GPU.
- If `pip install` reports a resolver conflict, use `-c environment/constraints-venv.txt` as shown in §4 —
  it pins the exact transitive set that was rehearsed; do not upgrade `scikit-learn` past 1.9.0 (the packed
  HistGradientBoosting model is a version-sensitive pickle) or `torch`/`torchvision` off the matching cu126
  pair.
- No compatibility issue requires or implies any change to the frozen model, its weights, or its inference
  logic; only environment/path fixes as above.

## 9 Final checklist

- [x] All `REPLACE_ME` fields have been replaced.
- [x] `manifest.yaml` matches the submitted files.
- [x] The package runs from a clean directory through `inference.sh` (rehearsed; see §7).
- [x] Every Selected Final Submission ID and its Kaggle-downloaded CSV are included and mapped to the
      corresponding model/configuration.
- [ ] `primary_verification_submission_id` is the ID confirmed by the Organizing Committee. *(entered by the
      team as `56094379`; awaiting organizer confirmation per process)*
- [x] External data, models, APIs, licences, and access requirements are declared.
- [x] `declarations/CUHK-X_Honor_Declaration.docx` is completed and signed, and the signed copy is included
      in `declarations/`.
- [x] No API keys, passwords, personal credentials, official raw challenge data, or unnecessary caches are
      included.
