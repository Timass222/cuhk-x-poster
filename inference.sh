#!/usr/bin/env bash
# Entry point for CUHK-X Small Model Track verification, team Timass.
#
#   bash inference.sh <data_dir> <output_csv> [submission_id]
#
# With two arguments this reproduces primary_verification_submission_id
# (56094379, pack PACK4_17U9, artifacts/model.pth). With a third argument
# equal to the other Selected Submission ID (56058031), it reproduces that
# submission instead (artifacts/model_pack3ira.pth). See
# submission/submission_info.yaml for the mapping.
#
# <data_dir> is the organizer-supplied directory holding the official
# label-free Testing data in the published Kaggle layout:
#   <prefix>/<clip_id>/{Depth_Color,IR,Thermal,Skeleton,IMU,Radar}/...
# Missing modalities per clip are tolerated (the affected branch is skipped
# by the fusion). No network access is used.
set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  echo "Usage: bash inference.sh <data_dir> <output_csv> [submission_id]" >&2
  exit 2
fi

PACKAGE_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
CALLER_PWD="$(pwd -W 2>/dev/null || pwd)"
abspath() { case "$1" in /*|[A-Za-z]:/*|[A-Za-z]:'\'*) printf '%s' "$1" ;; *) printf '%s/%s' "$CALLER_PWD" "$1" ;; esac; }

SRC="$(abspath "$1")"
OUT="$(abspath "$2")"
TARGET_SUBMISSION_ID="${3:-primary}"

if [ ! -e "$SRC" ]; then
  echo "Input data directory does not exist: $SRC" >&2
  exit 2
fi

case "$TARGET_SUBMISSION_ID" in
  primary|56094379) PACK="$PACKAGE_ROOT/artifacts/model.pth" ;;       # PACK4_17U9, submission_pack4_17u9_infer.csv
  56058031)         PACK="$PACKAGE_ROOT/artifacts/model_pack3ira.pth" ;; # submission_pack3ira_w20_infer.csv
  *)
    echo "Unknown submission_id: $TARGET_SUBMISSION_ID (expected 'primary', 56094379, or 56058031)" >&2
    exit 2
    ;;
esac

WORK="${WORK:-$PACKAGE_ROOT/work/stage2}"
mkdir -p "$WORK" "$(dirname -- "$OUT")"
cd "$PACKAGE_ROOT"

if [ -z "${PYTHON:-}" ]; then
  if [ -x ".venv/Scripts/python.exe" ]; then PYTHON=".venv/Scripts/python.exe"
  elif [ -x ".venv/bin/python" ]; then PYTHON=".venv/bin/python"
  else PYTHON="python3"; fi
fi
export PYTHONIOENCODING=utf-8
export CUNHX_META="$WORK/meta"
export CUNHX_CACHE="$WORK/cache"
mkdir -p "$CUNHX_META" "$CUNHX_CACHE"
LOG="$WORK/inference.log"
T0=$(date +%s)
step() { echo; echo "=========== $(date +%H:%M:%S) (+$(( $(date +%s) - T0 ))s) $1 ==========="; }
exec > >(tee -a "$LOG") 2>&1

echo "sample        : $SRC"
echo "submission_id : $TARGET_SUBMISSION_ID"
echo "pack          : $PACK ($(du -h "$PACK" | cut -f1))"
echo "work          : $WORK"
echo "python        : $PYTHON ($($PYTHON -c 'import torch;print("torch",torch.__version__,"cuda",torch.cuda.is_available())')))"

# Which branch inputs does THIS pack need? (union of spec.mods for branches whose weights are in the pack)
NEEDED=$($PYTHON - "$PACK" <<'PY'
import json, sys, numpy as np
z = np.load(sys.argv[1]); comps = {k.split("/")[0] for k in z.files if "/" in k}
mods = {"r34c5": ["Depth_ColorRGBcrop5", "IRcrop5"], "r34c2": ["Depth_ColorRGBcrop2", "IRcrop2"], "dir": ["Depth_Colorcrop", "IRcrop"], "thermal": ["Thermal"]}
if "branch_mods" in z.files: mods.update(json.loads(bytes(z["branch_mods"]).decode()))
spec = {"r34c5": {"prefix": "r34c5", "mods": mods["r34c5"]}, "r34c2": {"prefix": "r34c5", "mods": mods["r34c2"]},
        "dir_r2p1d": {"prefix": "dir", "mods": mods["dir"]}, "thermal_pl2": {"prefix": "thermal", "mods": mods["thermal"]}}
if "branch_spec" in z.files: spec.update(json.loads(bytes(z["branch_spec"]).decode()))
cfg = json.loads(bytes(z["fusion_cfg"]).decode()); W = cfg.get("weights", {}); names = set(cfg.get("names", spec))
need = sorted({m for n, v in spec.items() if n in names and W.get(n, 1.0) > 0 and f"{v['prefix']}/__bits__" in z.files for m in v["mods"]})
print(" ".join(need))
PY
)
echo "branch inputs needed by the pack: $NEEDED"
need() { case " $NEEDED " in *" $1 "*) return 0;; *) return 1;; esac; }

step "(1) meta: clip index, sessions (gap 1800 s), output paths"
$PYTHON src/infer/make_test_meta.py --src "$SRC" --meta "$CUNHX_META"

step "(2) frames 128px: Depth_Color, IR, Thermal"
$PYTHON src/data/preprocess/frames.py --mod Depth_Color --split test --zip "$SRC"
for M in IR Thermal; do
  if need $M; then $PYTHON src/data/preprocess/frames.py --mod $M --split test --zip "$SRC"; else echo "  $M 128-px frames not needed by this pack -> skipped"; fi
done

step "(3) motion boxes (Depth_Color) - dir branch + fallback"
$PYTHON src/data/preprocess/bbox.py --mod Depth_Color --split test

step "(4) YOLO person boxes on raw IR (weights from the pack) -> hybrid6"
$PYTHON src/infer/make_boxes.py --src "$SRC" --pack "$PACK"

step "(5) crops from the original frames: dir branch (motion box) + R34/ViT branches (hybrid6 box)"
need Depth_Colorcrop     && $PYTHON src/data/preprocess/crop_cache.py --mod Depth_Color --split test --zip "$SRC"
need IRcrop              && $PYTHON src/data/preprocess/crop_cache.py --mod IR --bbox-mod Depth_Color --split test --zip "$SRC"
need Depth_ColorRGBcrop6 && $PYTHON src/data/preprocess/crop_cache.py --mod Depth_Color --split test --zip "$SRC" --rgb --bbox-json "$CUNHX_META/bbox_hybrid6flat_test.json" --tag 6
need IRcrop6             && $PYTHON src/data/preprocess/crop_cache.py --mod IR --bbox-mod Depth_Color --split test --zip "$SRC" --bbox-json "$CUNHX_META/bbox_hybrid6flat_test.json" --tag 6
need Depth_ColorRGBcrop6h && $PYTHON src/data/preprocess/crop_cache.py --mod Depth_Color --split test --zip "$SRC" --rgb --bbox-json "$CUNHX_META/bbox_hybrid6flat_test.json" --tag 6h --size 224
need IRcrop6h             && $PYTHON src/data/preprocess/crop_cache.py --mod IR --bbox-mod Depth_Color --split test --zip "$SRC" --bbox-json "$CUNHX_META/bbox_hybrid6flat_test.json" --tag 6h --size 224
for M in Depth_ColorRGBcrop5 IRcrop5 Depth_ColorRGBcrop2 IRcrop2; do
  need $M && { echo "ERROR: pack needs $M (legacy crop tag) which this script does not build" >&2; exit 2; }
done
true

step "(6) IMU tensors/features + skeleton sequences"
$PYTHON src/data/preprocess/build_caches.py --test-zip "$SRC"

step "(7) pack inference: branches -> fusion -> session decoder -> CSV"
$PYTHON src/infer/pack_infer.py --pack "$PACK" --out "$OUT"

step "DONE: $OUT ($(wc -l < "$OUT") lines incl. header), total $(( $(date +%s) - T0 ))s"
