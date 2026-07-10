"""
One-time script: split SegmentedForests LAZ files by the embedded Split field.

Input  (default):  data/plot_01.laz ... data/plot_14.laz
Output (default):  data/forest/raw/train/plot_01.laz ... plot_14.laz  (Split == 0)
                   data/forest/raw/val/plot_01.laz   ... plot_14.laz  (Split == 1)
                   data/forest/raw/test/plot_01.laz  ... plot_14.laz  (Split == 2)

Usage:
    python scripts/prepare_forest_data.py
    python scripts/prepare_forest_data.py --src /custom/src --dst /custom/dst
"""

import argparse
import os
import sys

import laspy
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DEFAULT = os.path.join(ROOT, "data")
DST_DEFAULT = os.path.join(ROOT, "data", "forest", "raw")

SPLIT_MAP = {0: "train", 1: "val", 2: "test"}


def split_plot(src_path: str, dst_dir: str, plot_name: str) -> None:
    print(f"  {os.path.basename(src_path)}", end=" ... ", flush=True)
    las = laspy.read(src_path)
    split_vals = np.array(las["Split"])
    print(f"{len(split_vals):,} pts")

    for split_val, stage in SPLIT_MAP.items():
        mask = split_vals == split_val
        n = int(mask.sum())
        if n == 0:
            print(f"    [{stage:>5}]  0 pts — skipped")
            continue

        out_path = os.path.join(dst_dir, stage, f"{plot_name}.laz")

        # Copy header (preserves scales, offsets, extra-byte definitions)
        new_las = laspy.LasData(header=las.header)
        new_las.points = las.points[mask]
        new_las.write(out_path)

        size_mb = os.path.getsize(out_path) / 1e6
        print(f"    [{stage:>5}]  {n:>10,} pts  →  {os.path.basename(out_path)}  ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=SRC_DEFAULT,
                        help="Directory that contains plot_01.laz … plot_14.laz")
    parser.add_argument("--dst", default=DST_DEFAULT,
                        help="Output raw dir (train/val/test sub-dirs created here)")
    args = parser.parse_args()

    src = os.path.abspath(args.src)
    dst = os.path.abspath(args.dst)

    for stage in SPLIT_MAP.values():
        os.makedirs(os.path.join(dst, stage), exist_ok=True)

    laz_files = sorted(f for f in os.listdir(src) if f.startswith("plot_") and f.endswith(".laz"))
    if not laz_files:
        print(f"ERROR: no plot_*.laz files found in {src}")
        sys.exit(1)

    print(f"\nFound {len(laz_files)} plot(s) in {src}")
    print(f"Writing to {dst}\n")

    for fname in laz_files:
        split_plot(os.path.join(src, fname), dst, os.path.splitext(fname)[0])
        print()

    print("=== Summary ===")
    for stage in SPLIT_MAP.values():
        stage_dir = os.path.join(dst, stage)
        files = sorted(os.listdir(stage_dir))
        total_mb = sum(os.path.getsize(os.path.join(stage_dir, f)) for f in files) / 1e6
        print(f"  {stage:>5}/  {len(files)} files  {total_mb:.0f} MB total")


if __name__ == "__main__":
    main()
