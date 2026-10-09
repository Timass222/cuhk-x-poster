# Timestamps are a free sensor

Team Timass · CUHK-X Challenge, Small Model Track · UbiComp / ISWC 2026 Grand Finals

**▶ Video: [what the session decoder changes, step by step](https://timass222.github.io/cuhk-x-poster/)**

- **Task.** 40 daily activities recognised from depth, infrared, skeleton and IMU clips (no RGB), tested on
  people never seen in training.
- **Model.** Five small branches with late fusion, in one 94 MiB file: VideoMAEv2 ViT-B/16 on Depth + IR person
  crops, ViT-S/16 on IR, CTR-GCN on 2-D skeletons, gradient boosting and a small network on IMU.
- **Session decoder.** Clip start times chain the test clips into sessions. A second-order HMM
  (forward–backward) labels the whole session offline, using transition tables counted on the training lists
  and no test labels.

## Results

Hidden test, all 405 clips (Kaggle late submissions, public + private); decoder steps added in this order.

| Step | Accuracy | Correct |
|---|---|---|
| Clips alone (fused ensemble) | 75.8 % | 307 |
| + one action back (first-order transitions) | 83.0 % | 336 |
| + two actions back (second-order transitions) | 84.0 % | 340 |
| + duration prior, repeat pooling, one action per list | 85.4 % | 346 |
| + session adaptation (prototypes, EM on the session) | 86.4 % | 350 |

Official private-leaderboard score of the Selected submission: 0.86764 (submission 56094379).

## The decoder in one formula

For every clip, each of the 40 classes gets

```
score(c) ∝ clip(c)^0.85 × T(a, b → c)
T(a, b → c) ∝ (0.75 · P2(c | a, b) + 0.25 · P1(c | b))^0.75      (renormalised over c)
```

`clip(c)` is the fused per-clip probability; `P1`, `P2` are transition frequencies counted on the training lists
given the last one or two actions (an interpolated second-order HMM, Jelinek–Mercer style). The power 0.75 and the
renormalisation flatten the tables, so transitions never seen in training keep a small chance. Forward–backward runs
this over the whole session, so it also averages over uncertain earlier clips and takes the next clip into account.

Worked example (step 1 of the video, one action back only, so `T ∝ P1^0.75`): after *Peel fruits*, *Eat food* has
P1 = 0.545 → 0.545^0.75 = 0.634 → 0.34 after renormalising over the 40 classes. The clip alone gives *Eat food*
0.116^0.85 = 0.16 and *Use a mobile phone* 0.382^0.85 = 0.44. Products 0.16 × 0.34 = 0.054 and 0.44 × 0.011 = 0.005;
normalised over all 40 classes that is 0.74 vs 0.07. Full forward–backward gives 0.72 vs 0.07.

Worked example with a transition never seen in training (step 2 of the video, left panel of the poster): after
*Sit down → Tap the keyboard* (18 training lists) *Write* never came next, so the pseudo-count 0.1 gives
P2(Write) = (0 + 0.1) / (18 + 40 · 0.1) = 0.0045, against P2(Check the time) = 0.595. Mixed with one action back:

| | 0.75 · P2 | + 0.25 · P1 | = mix | ^0.75 | T after renormalising |
|---|---|---|---|---|---|
| Check the time | 0.447 | 0.089 (P1 0.355) | 0.535 | 0.626 | **0.347** |
| Write | 0.003 | 0.039 (P1 0.157) | 0.043 | 0.094 | **0.052** |

Never seen in training means unlikely, not impossible. The clip says Write 0.226^0.85 = 0.28 vs Check the time
0.069^0.85 = 0.10 (about 3 : 1); products 0.10 × 0.347 = 0.036 vs 0.28 × 0.052 = 0.015, so Check the time wins.
To keep Write, the clip would have needed about 9 : 1. Full forward–backward gives 0.29 vs 0.22, because it also
weighs the uncertainty of the earlier clips.

If a pair (a, b) never occurred in training at all, P2 is uniform (1/40) and T falls back to a flattened first-order
table. On the hidden test this applies to 23 of the 146 clips that have two predecessors in their list.

**Adaptation to the session (+1.0 point, no test labels).** After the first decoder pass, every session gets class
prototypes: the mean ViT-B feature (fc_norm, 768-d) of its clips with each decoded label, shrunk toward the training
prototype of that class with weight 3 clips. The cosine of each clip to the 40 prototypes, through a softmax at
temperature 0.05, multiplies the fused probabilities with power 0.3, and the session is decoded again. Then transitions
counted on the decoded test sessions are mixed into the training tables at 30 % (one EM step) for a third pass. On the
hidden test the prototypes change 8 answers and EM 3 more, a net +4 correct clips (346 → 350); in 4 of the 8 cases the
prototypes restore the clip's own answer that the transitions had overruled. Prototypes rely on repeated clips of the
same action within a session, so we report this step separately from the learned transitions.
Code: [src/infer/pack_infer.py](src/infer/pack_infer.py), constants in `fusion_cfg` of `model.pth`.

## Run it

1. Download `model.pth` from the [latest release](https://github.com/Timass222/cuhk-x-poster/releases/latest)
   into `artifacts/model.pth`.
2. Create `.venv` in the repo root (inference.sh uses it, or the interpreter in `$PYTHON`) and install the pinned
   environment, or build `environment/Dockerfile`:
   ```
   pip install --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.org/simple \
       -r environment/requirements.txt -c environment/constraints-venv.txt
   ```
3. `bash inference.sh <test_dir> <output.csv>` reproduces `submission/final_submission_1.csv` (405/405).
   `<test_dir>` is the folder that contains `small_model_track_test/`, as in the Kaggle zip.

Only the primary pack is released. The secondary submission (`inference.sh … 56058031`) needs
`model_pack3ira.pth`, which is not published.

Full reproduction guide: [REPRODUCE.md](REPRODUCE.md). Method and numbers: [docs/tech_report.md](docs/tech_report.md).

## Data and third-party components

- The CUHK-X dataset is not included; it is distributed by the challenge organisers
  ([paper](https://arxiv.org/abs/2512.07136)).
- YOLO11n person detector by Ultralytics, AGPL-3.0: `src/infer/yolo11n.pt` and a copy inside `model.pth`.
- Backbone initialisations: VideoMAEv2 ViT-B/16 and ViT-S/16 (OpenGVLab), PYSKL CTR-GCN (NTU-60). Their
  licences apply to the derived weights.

## Licence

Code: MIT ([LICENSE](LICENSE)). The YOLO11n detector weights are AGPL-3.0 (Ultralytics) and are not covered by
the MIT licence.
