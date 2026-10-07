#!/usr/bin/env bash
# Wait for the browser to finish the 8 train volumes, move them into
# data/raw/ under their canonical names, extract the light modalities.
set -u
cd "$(dirname "$0")/../.."

echo "[intake] waiting for 8 completed volumes (HAR-0**.z0N without .crdownload)"
for i in $(seq 1 120); do
  n=$(ls HAR-0*.z0[1-8] 2>/dev/null | grep -vc crdownload || true)
  [ "$n" -ge 8 ] && break
  sleep 30
done
n=$(ls HAR-0*.z0[1-8] 2>/dev/null | grep -vc crdownload || true)
if [ "$n" -lt 8 ]; then
  echo "[intake] TIMEOUT: only $n volumes complete after 60 min"; exit 2
fi

echo "[intake] all 8 volumes complete; moving to data/raw/"
for f in HAR-0*.z0[1-8]; do
  ext="${f##*.}"                      # z01..z08 -- the extension is the truth
  tgt="data/raw/HAR.${ext}"
  if [ -e "$tgt" ]; then echo "  skip $f -> $tgt (exists)"; continue; fi
  mv -n "$f" "$tgt" && echo "  $f -> $tgt"
done

ls -la data/raw/HAR.z0* data/raw/HAR.zip

echo "[intake] extracting Skeleton IMU Radar (light, ~0.31 GB)"
"C:/Program Files/7-Zip/7z.exe" x data/raw/HAR.zip \
  "HAR/data/Skeleton/*" "HAR/data/IMU/*" "HAR/data/Radar/*" \
  -odata/extracted -y -bsp0 -bso0
echo "[intake] 7z exit: $?"

echo "[intake] extracted file counts:"
for m in Skeleton IMU Radar; do
  c=$(find "data/extracted/HAR/data/$m" -type f 2>/dev/null | wc -l)
  echo "  $m: $c files"
done
echo "[intake] done"
