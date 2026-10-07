"""Selective extraction from the merged HAR archive.

Modality is the TOP level of the training hierarchy:

    HAR/data/<modality>/<action>/<user>/<trial>/<files>

so each modality can be unpacked on its own. Start with the cheap ones
(Skeleton, IMU, Radar) -- they are small, they train on a laptop CPU, and
they are the modalities the public 0.716 notebook ignores entirely.

Merging the split volumes first (all volumes must sit in one directory):

    zip -s 0 data/raw/HAR.zip --out data/raw/HAR_full.zip     # zip >= 3.0
    # Windows: 7-Zip opens data/raw/HAR.zip directly when z01..z08 are present.

Then:

    python code/data/extract.py Skeleton IMU
    python code/data/extract.py Depth_Color --dest data/extracted

Peak disk: 44.6 GB (archive) + whatever you extract. Delete the archive only
after every modality you need is on disk -- re-downloading costs 44.6 GB.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def find_seven_zip() -> str | None:
    for candidate in ("7z", "7za", r"C:\Program Files\7-Zip\7z.exe"):
        if shutil.which(candidate) or Path(candidate).is_file():
            return candidate
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("modalities", nargs="+",
                        help="e.g. Skeleton IMU Radar Depth_Color IR Thermal")
    parser.add_argument("--archive", type=Path, default=Path("data/raw/HAR.zip"))
    parser.add_argument("--dest", type=Path, default=Path("data/extracted"))
    args = parser.parse_args()

    if not args.archive.is_file():
        print(f"not found: {args.archive}", file=sys.stderr)
        return 1

    seven_zip = find_seven_zip()
    if seven_zip is None:
        print("7-Zip not found. Install it, or use:\n"
              "  unzip data/raw/HAR_full.zip 'HAR/data/<Modality>/*' -d data/extracted",
              file=sys.stderr)
        return 1

    args.dest.mkdir(parents=True, exist_ok=True)
    for modality in args.modalities:
        pattern = f"HAR/data/{modality}/*"
        print(f"\n=== {modality} ===")
        command = [seven_zip, "x", str(args.archive), pattern,
                   f"-o{args.dest}", "-y"]
        result = subprocess.run(command)
        if result.returncode != 0:
            print(f"7-Zip failed for {modality} (exit {result.returncode})",
                  file=sys.stderr)
            return result.returncode
    print("\ndone. next: python code/data/index.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
