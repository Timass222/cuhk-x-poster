"""Staged download of the CUHK-X Small Model Track dataset from Hugging Face.

The repo is GATED: accept the terms at
    https://huggingface.co/datasets/Kevin-Pal/CUHK-X_Small_Model_Track
then authenticate once with `huggingface-cli login` (or pass --token).

Stages, cheapest first -- do not download everything up front:

  probe  1.67 GB  HAR.zip only, the final volume of the split archive.
                  Its central directory lists every file in all 44.6 GB.
                  Feed it to code/data/inventory.py. Answers the user-overlap
                  question and sizes every modality before you commit disk.

  test   2.79 GB  small_model_track_test.zip + test.csv + sample_submission.csv
                  Needed to build the per-clip modality availability matrix.

  train    43 GB  HAR.z01 ... HAR.z08, the remaining volumes.
                  Only needed once you actually train. Run it in the background.

Usage
-----
    python code/data/download.py probe
    python code/data/download.py test
    python code/data/download.py train        # long; resumable
    python code/data/download.py all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ID = "Kevin-Pal/CUHK-X_Small_Model_Track"
PREFIX = "Small-Model-Track"

STAGES: dict[str, list[str]] = {
    "probe": [f"{PREFIX}/Training/data/HAR.zip"],
    "test": [
        f"{PREFIX}/Testing/data/small_model_track_test.zip",
        f"{PREFIX}/Testing/test_file/test.csv",
        f"{PREFIX}/Testing/test_file/sample_submission.csv",
        f"{PREFIX}/class_mapping.csv",
    ],
    "train": [f"{PREFIX}/Training/data/HAR.z{i:02d}" for i in range(1, 9)],
}
STAGES["all"] = STAGES["probe"] + STAGES["test"] + STAGES["train"]

APPROX_GB = {"probe": 1.67, "test": 2.79, "train": 42.9, "all": 47.4}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=sorted(STAGES))
    parser.add_argument("--out", type=Path, default=Path("data/raw"))
    parser.add_argument("--token", default=None, help="HF token; omit if logged in")
    args = parser.parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("pip install huggingface_hub", file=sys.stderr)
        return 1

    files = STAGES[args.stage]
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"stage '{args.stage}': {len(files)} file(s), ~{APPROX_GB[args.stage]} GB")
    print(f"destination: {args.out.resolve()}\n")

    for number, filename in enumerate(files, start=1):
        print(f"[{number}/{len(files)}] {filename}")
        path = hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=filename,
            local_dir=args.out,
            token=args.token,
        )
        # Flatten: keep archive volumes side by side so 7-Zip can span them.
        source = Path(path)
        flat = args.out / source.name
        if source.resolve() != flat.resolve():
            flat.parent.mkdir(parents=True, exist_ok=True)
            if flat.exists():
                flat.unlink()
            source.replace(flat)
        print(f"    -> {flat}")

    if args.stage in ("probe", "all"):
        print("\nnext:  python code/data/inventory.py data/raw/HAR.zip "
              "--json data/meta/inventory.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
