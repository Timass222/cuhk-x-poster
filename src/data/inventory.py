"""Inventory the full HAR training archive from its LAST volume only.

Why this works
--------------
`HAR.z01 ... HAR.z08 + HAR.zip` is a PKZIP split archive (44.6 GB total).
The End-of-Central-Directory record and the entire central directory live at
the very end of the archive, i.e. inside `HAR.zip` (1.67 GB). That central
directory lists *every* file in all volumes -- name and uncompressed size.

`unzip -l` / 7-Zip refuse to open a multi-volume archive when volumes are
missing, so we parse the central-directory headers out of the raw bytes
instead. That needs only the final volume.

What it answers
---------------
1. Which modalities exist, how many files each, how many bytes each
   -> plan selective extraction instead of unpacking 44.6 GB.
2. Which user IDs appear in TRAINING data.
   -> The challenge site announces train = users 1-9,16-24 and
      test = users 10,11,25,26. The HF dataset card's quick-start example,
      however, shows `HAR/data/IR/0_Wash_face/user10/4-2-1/` -- user10 inside
      the *training* archive. If users 10/11/25/26 really are present here,
      the public leaderboard's 0.98 cluster is explained without any leak:
      those teams train on the very subjects the test set is drawn from.
      This is the single highest-value question about this competition.
3. Class list and per-class / per-user clip counts.

Usage
-----
    python code/data/inventory.py data/raw/HAR.zip
    python code/data/inventory.py data/raw/HAR.zip --json data/meta/inventory.json
"""

from __future__ import annotations

import argparse
import collections
import json
import mmap
import re
import struct
import sys
from pathlib import Path

CENTRAL_DIR_SIG = b"PK\x01\x02"

# HAR/data/<modality>/<action>/<user>/<trial>/<file>
ENTRY_RE = re.compile(
    r"HAR/data/(?P<modality>[^/]+)/(?P<action>[^/]+)/(?P<user>[^/]+)/(?P<trial>[^/]+)/"
)

# Announced split, from the challenge website.
ANNOUNCED_TRAIN = set(range(1, 10)) | set(range(16, 25))
ANNOUNCED_TEST = {10, 11, 25, 26}


def iter_central_directory(path: Path):
    """Yield (filename, uncompressed_size) for every central-directory entry."""
    with open(path, "rb") as handle:
        mm = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            position = 0
            while True:
                index = mm.find(CENTRAL_DIR_SIG, position)
                if index < 0:
                    return
                position = index + 4
                if index + 46 > len(mm):
                    return
                try:
                    size = struct.unpack_from("<I", mm, index + 24)[0]
                    name_length = struct.unpack_from("<H", mm, index + 28)[0]
                except struct.error:
                    continue
                if not 0 < name_length < 4096:
                    continue  # not a real header, just a byte coincidence
                raw = mm[index + 46 : index + 46 + name_length]
                if len(raw) != name_length:
                    continue
                yield raw.decode("utf-8", "replace"), size
        finally:
            mm.close()


def user_number(user: str) -> int | None:
    match = re.search(r"(\d+)", user)
    return int(match.group(1)) if match else None


def build_inventory(path: Path) -> dict:
    modality_files = collections.Counter()
    modality_bytes = collections.Counter()
    users = collections.Counter()
    actions = collections.Counter()
    clips: set[tuple[str, str, str, str]] = set()
    unmatched = 0
    total = 0

    for name, size in iter_central_directory(path):
        total += 1
        match = ENTRY_RE.search(name)
        if match is None:
            unmatched += 1
            continue
        modality = match.group("modality")
        modality_files[modality] += 1
        modality_bytes[modality] += size
        users[match.group("user")] += 1
        actions[match.group("action")] += 1
        clips.add(
            (modality, match.group("action"), match.group("user"), match.group("trial"))
        )

    numbered = sorted(
        {n for user in users if (n := user_number(user)) is not None}
    )
    return {
        "archive": str(path),
        "central_directory_entries": total,
        "unmatched_entries": unmatched,
        "modalities": {
            m: {"files": modality_files[m], "bytes": modality_bytes[m]}
            for m in sorted(modality_files, key=lambda k: -modality_bytes[k])
        },
        "users": sorted(users),
        "user_numbers": numbered,
        "actions": sorted(actions),
        "clips": len(clips),
    }


def report(inv: dict) -> None:
    print(f"central-directory entries : {inv['central_directory_entries']:,}")
    print(f"unmatched (non HAR/data/) : {inv['unmatched_entries']:,}")
    print(f"distinct clips            : {inv['clips']:,}")
    print(f"action classes            : {len(inv['actions'])}")

    print("\nMODALITIES  (extract selectively: 7z x HAR_full.zip \"HAR/data/<M>/*\")")
    print(f"  {'modality':<16}{'files':>12}{'size':>12}")
    for modality, stats in inv["modalities"].items():
        print(f"  {modality:<16}{stats['files']:>12,}{stats['bytes'] / 1e9:>11.2f} GB")

    present = set(inv["user_numbers"])
    print(f"\nUSERS PRESENT IN TRAINING ARCHIVE ({len(present)}):")
    print(f"  {sorted(present)}")

    leaked = sorted(present & ANNOUNCED_TEST)
    missing = sorted(ANNOUNCED_TRAIN - present)
    extra = sorted(present - ANNOUNCED_TRAIN - ANNOUNCED_TEST)

    print("\n" + "=" * 68)
    if leaked:
        print("!!  ANNOUNCED TEST USERS FOUND IN THE TRAINING ARCHIVE:", leaked)
        print("!!  The cross-subject split is not what the website advertises.")
        print("!!  Training on these subjects makes the public test SAME-subject,")
        print("!!  which plausibly explains the 0.98 cluster on the leaderboard.")
        print("!!  Decide deliberately whether to use them, then rebuild the CV split.")
    else:
        print("OK  No announced test users (10, 11, 25, 26) in the training archive.")
        print("OK  The split is genuinely cross-subject; the HF card example was sloppy.")
        print("OK  The 0.98 cluster needs a different explanation.")
    print("=" * 68)

    if missing:
        print(f"\nnote: announced train users absent from archive: {missing}")
    if extra:
        print(f"note: users outside the announced split entirely: {extra}")

    print(f"\nfirst 10 action classes: {inv['actions'][:10]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("archive", type=Path, help="path to HAR.zip (final volume)")
    parser.add_argument("--json", type=Path, help="also write the inventory as JSON")
    args = parser.parse_args()

    if not args.archive.is_file():
        print(f"not found: {args.archive}", file=sys.stderr)
        return 1

    inventory = build_inventory(args.archive)
    report(inventory)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(inventory, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
