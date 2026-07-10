"""
Universal LAZ/TXT splitter with optional SPT preprocessing.

For each input file the script:
  • Accepts .laz or .txt inputs.  A .txt file is automatically converted to
    .laz (same directory, same stem) before splitting.
  • Detects whether a "Split" extra-byte field is present.
  • If present  → routes points to train / val / test using the embedded
                  split value (0 / 1 / 2), identical to prepare_forest_data.py.
  • If absent   → divides the point cloud spatially along its longest
                  horizontal axis into contiguous train / val / test bands.

With --preprocess the script also runs SPT preprocessing to convert the
split LAZ files into ready-to-use h5 tile files.  Everything is kept under
data/Inference/ so the existing data/forest/ tree is never modified.

TXT format
----------
  Expected columns: x  y  z  r  g  b  z1   (any capitalisation)
  Delimiter: auto-detected (tab, comma, or whitespace).
  Header:    auto-detected (first line is skipped when its first token is
             not a valid float).
  RGB scale: auto-detected from the channel maximum:
             max <= 1.5   → float [0–1]    (×65535 → uint16)
             max <= 255.5 → uint8 [0–255]  (×257   → uint16)
             otherwise    → uint16 [0–65535] (used as-is)
  z1:        stored as a float32 extra-byte in the LAZ; ignored by SPT.
  Memory:    reading a 27 M-point file uses roughly 1.5 GB RAM; files
             larger than ~50 M points may exhaust available memory.

Output layout
-------------
  Without --preprocess (split only):
    <dst>/train/<filename>.laz
    <dst>/val/<filename>.laz
    <dst>/test/<filename>.laz

  With --preprocess (default dst = data/Inference/raw/):
    data/Inference/raw/train/<filename>.laz     ← split LAZ files
    data/Inference/raw/val/<filename>.laz
    data/Inference/raw/test/<filename>.laz
    data/Inference/processed/<hash>/train/      ← h5 tiles (SPT format)
    data/Inference/processed/<hash>/val/
    data/Inference/processed/<hash>/test/

Usage
-----
    # TXT input — split + preprocess in one step:
    python scripts/prepare_laz.py data/Inference/cloud.txt --preprocess

    # LAZ input — split + preprocess:
    python scripts/prepare_laz.py data/Inference/survey_A.laz --preprocess

    # Custom split ratio (60 / 20 / 20 %):
    python scripts/prepare_laz.py *.laz --preprocess --ratio 0.6 0.2 0.2

    # Force Y-axis split:
    python scripts/prepare_laz.py survey.laz --preprocess --axis y

NOTE — small plots
    If a plot's extent per third is less than ~10 m, SPT's default pc_tiling=2
    will subdivide further and may create near-empty tiles.  Set pc_tiling=1
    in configs/datamodule/semantic/forest.yaml for small inputs.
"""

import argparse
import os
import sys
import logging
from pathlib import Path

import laspy
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

log = logging.getLogger(__name__)
logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)

DST_DEFAULT           = str(REPO / "data" / "forest"    / "raw")
DST_INFERENCE_DEFAULT = str(REPO / "data" / "Inference" / "raw")

SPLIT_MAP = {0: "train", 1: "val", 2: "test"}
STAGES    = ("train", "val", "test")

MIN_PTS_WARN = 10_000


# ─────────────────────────────────────────────────────────────────────────────
# TXT → LAZ conversion
# ─────────────────────────────────────────────────────────────────────────────

def _detect_sep(filepath: str, data_line_idx: int) -> str:
    """Return the pandas sep string inferred from one data line."""
    with open(filepath) as fh:
        for _ in range(data_line_idx):
            fh.readline()
        line = fh.readline()
    if '\t' in line:
        return '\t'
    if ',' in line:
        return ','
    return r'\s+'


def txt_to_laz(txt_path: str) -> str:
    """Convert a .txt point cloud (x, y, z, r, g, b, z1) to a .laz file.

    Auto-detects delimiter, header, and RGB scale.
    z1 is stored as a float32 extra-byte (SPT ignores unknown extra-bytes).

    Returns the path to the written .laz file (same dir, same stem as input).
    Memory note: ~1.5 GB RAM for a 27 M-point file.
    """
    import pandas as pd

    laz_path = os.path.splitext(txt_path)[0] + '.laz'

    # ── Header auto-detection ─────────────────────────────────────────────────
    with open(txt_path) as fh:
        first_line = fh.readline().strip()
    first_token = first_line.split()[0] if first_line else ''
    try:
        float(first_token)
        has_header = False
    except ValueError:
        has_header = True

    # ── Separator auto-detection ──────────────────────────────────────────────
    sep = _detect_sep(txt_path, data_line_idx=1 if has_header else 0)
    sep_label = 'tab' if sep == '\t' else 'comma' if sep == ',' else 'whitespace'

    # ── Read ──────────────────────────────────────────────────────────────────
    log.info(
        f"Reading {os.path.basename(txt_path)}  "
        f"(sep={sep_label}, header={'yes' if has_header else 'no'}) …"
    )
    default_cols = ['x', 'y', 'z', 'r', 'g', 'b', 'z1']
    engine = 'python' if sep == r'\s+' else 'c'
    df = pd.read_csv(
        txt_path,
        sep=sep,
        engine=engine,
        header=0 if has_header else None,
        names=None if has_header else default_cols,
    )
    df.columns = [c.lower() for c in df.columns]
    n_pts = len(df)
    log.info(f"  {n_pts:,} points")

    x  = df['x'].to_numpy(dtype=np.float64)
    y  = df['y'].to_numpy(dtype=np.float64)
    z  = df['z'].to_numpy(dtype=np.float64)
    r  = df['r'].to_numpy(dtype=np.float64)
    g  = df['g'].to_numpy(dtype=np.float64)
    b  = df['b'].to_numpy(dtype=np.float64)
    z1 = df['z1'].to_numpy(dtype=np.float32)
    del df

    # ── RGB auto-scale ────────────────────────────────────────────────────────
    rgb_max = float(max(r.max(), g.max(), b.max()))
    if rgb_max <= 1.5:
        log.info(f"  RGB: float [0–1] (max={rgb_max:.3f}) → ×65535 → uint16")
        r_u16 = np.clip(r * 65535, 0, 65535).astype(np.uint16)
        g_u16 = np.clip(g * 65535, 0, 65535).astype(np.uint16)
        b_u16 = np.clip(b * 65535, 0, 65535).astype(np.uint16)
    elif rgb_max <= 255.5:
        log.info(f"  RGB: uint8 [0–255] (max={rgb_max:.0f}) → ×257 → uint16")
        r_u16 = np.clip(r * 257, 0, 65535).astype(np.uint16)
        g_u16 = np.clip(g * 257, 0, 65535).astype(np.uint16)
        b_u16 = np.clip(b * 257, 0, 65535).astype(np.uint16)
    else:
        log.info(f"  RGB: uint16 [0–65535] (max={rgb_max:.0f}) — used as-is")
        r_u16 = np.clip(r, 0, 65535).astype(np.uint16)
        g_u16 = np.clip(g, 0, 65535).astype(np.uint16)
        b_u16 = np.clip(b, 0, 65535).astype(np.uint16)

    # ── Guard z1 against non-finite values ────────────────────────────────────
    bad = ~np.isfinite(z1)
    if bad.any():
        log.warning(f"  z1: {int(bad.sum())} non-finite value(s) replaced with 0.0")
        z1[bad] = 0.0

    # ── Build and write LAZ (point format 2 = x/y/z + RGB) ───────────────────
    header = laspy.LasHeader(point_format=2, version="1.2")
    header.add_extra_dim(laspy.ExtraBytesParams(name='z1', type=np.float32))
    header.offsets = np.array([x.min(), y.min(), z.min()])
    header.scales  = np.array([0.001, 0.001, 0.001])

    las       = laspy.LasData(header=header)
    las.x     = x
    las.y     = y
    las.z     = z
    las.red   = r_u16
    las.green = g_u16
    las.blue  = b_u16
    las['z1'] = z1

    las.write(laz_path)
    size_mb = os.path.getsize(laz_path) / 1e6
    log.info(f"  → {os.path.basename(laz_path)}  ({n_pts:,} pts, {size_mb:.0f} MB)")
    return laz_path


# ─────────────────────────────────────────────────────────────────────────────
# LAZ split helpers
# ─────────────────────────────────────────────────────────────────────────────

def _write_split(las: laspy.LasData, mask: np.ndarray,
                 out_path: str, stage: str) -> int:
    n = int(mask.sum())
    if n == 0:
        print(f"    [{stage:>5}]  0 pts — skipped")
        return 0
    new_las = laspy.LasData(header=las.header)
    new_las.points = las.points[mask]
    new_las.write(out_path)
    size_mb = os.path.getsize(out_path) / 1e6
    warn = "  ⚠  very few points — consider pc_tiling=1" if n < MIN_PTS_WARN else ""
    print(f"    [{stage:>5}]  {n:>12,} pts  →  {os.path.basename(out_path)}"
          f"  ({size_mb:.1f} MB){warn}")
    return n


def _split_by_bbox(las: laspy.LasData, ratio: tuple, axis: str):
    """Return boolean masks (train, val, test) from a bounding-box spatial split."""
    x  = np.asarray(las.x, dtype=np.float64)
    y  = np.asarray(las.y, dtype=np.float64)
    dx = float(x.max() - x.min())
    dy = float(y.max() - y.min())

    use_x      = (dx >= dy) if axis == "auto" else (axis.lower() == "x")
    coord      = x if use_x else y
    axis_label = "X" if use_x else "Y"
    extent     = float(coord.max() - coord.min())
    cmin       = float(coord.min())

    r0, r1, _ = ratio
    b1 = cmin + extent * r0
    b2 = cmin + extent * (r0 + r1)

    return (coord < b1), (coord >= b1) & (coord < b2), (coord >= b2), axis_label, extent


# ─────────────────────────────────────────────────────────────────────────────
# Per-file processing
# ─────────────────────────────────────────────────────────────────────────────

def process_file(src_path: str, dst_dir: str, ratio: tuple, axis: str) -> None:
    plot_name   = os.path.splitext(os.path.basename(src_path))[0]
    las         = laspy.read(src_path)
    n_total     = len(las.points)
    extra_names = set(las.point_format.extra_dimension_names)

    print(f"\n  {os.path.basename(src_path)}  ({n_total:,} pts)")

    if "Split" in extra_names:
        print("    Mode: Split field detected — using embedded split values")
        split_vals = np.asarray(las["Split"])
        for split_val, stage in SPLIT_MAP.items():
            out_path = os.path.join(dst_dir, stage, f"{plot_name}.laz")
            _write_split(las, split_vals == split_val, out_path, stage)
    else:
        train_mask, val_mask, test_mask, axis_label, extent = \
            _split_by_bbox(las, ratio, axis)
        r0, r1, r2 = ratio
        print(f"    Mode: no Split field — spatial split along {axis_label}-axis"
              f"  (extent {extent:.1f} m)  ratio {r0:.3f}/{r1:.3f}/{r2:.3f}")
        for mask, stage in zip((train_mask, val_mask, test_mask), STAGES):
            out_path = os.path.join(dst_dir, stage, f"{plot_name}.laz")
            _write_split(las, mask, out_path, stage)


# ─────────────────────────────────────────────────────────────────────────────
# SPT preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def _run_preprocessing() -> None:
    """Trigger SPT preprocessing for all splits under data/Inference/raw/.

    Processed h5 tiles are written to data/Inference/processed/.
    Uses InferenceForestDataModule so data/forest/ is never touched.
    """
    import pyrootutils
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf
    import hydra.utils as hu

    pyrootutils.setup_root(str(REPO), indicator=".project-root", pythonpath=True)

    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)

    GlobalHydra.instance().clear()
    with initialize_config_dir(
            config_dir=str(REPO / "configs"), version_base="1.2"):
        cfg = compose(
            config_name="eval",
            overrides=[
                "experiment=semantic/forest",
                "trainer=gpu",
                "logger=csv",
                "extras.print_config=false",
                "datamodule._target_=src.datamodules.forest.InferenceForestDataModule",
            ],
        )

    datamodule = hu.instantiate(cfg.datamodule)

    log.info("─── Preprocessing train + val splits ───")
    datamodule.setup("fit")

    log.info("─── Preprocessing test split ───")
    datamodule.setup("predict")

    proc_dir = str(REPO / "data" / "Inference" / "processed")
    log.info(f"Preprocessing complete.  h5 files → {proc_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Split LAZ or TXT point clouds into train/val/test, "
            "optionally preprocess for SPT.\n"
            "TXT files are converted to LAZ automatically (same dir, same stem).\n"
            "Memory note: reading a 27 M-point TXT file uses ~1.5 GB RAM."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "inputs", nargs="+", metavar="FILE",
        help="One or more input .laz or .txt files",
    )
    parser.add_argument(
        "--dst", default=None, metavar="DIR",
        help="Output root for split LAZ files.  Defaults to data/Inference/raw/ "
             "when --preprocess is given, data/forest/raw/ otherwise.",
    )
    parser.add_argument(
        "--ratio", nargs=3, type=float, default=[1/3, 1/3, 1/3],
        metavar=("TRAIN", "VAL", "TEST"),
        help="Spatial split ratios when no Split field is present "
             "(default: 0.333 0.333 0.333). Must sum to 1.",
    )
    parser.add_argument(
        "--axis", default="auto", choices=["auto", "x", "y"],
        help="Axis for spatial split when no Split field is present "
             "(default: auto = longest horizontal axis)",
    )
    parser.add_argument(
        "--preprocess", action="store_true",
        help="After splitting, run SPT preprocessing to create h5 tiles under "
             "data/Inference/processed/  (requires a GPU and CUDA environment).",
    )
    args = parser.parse_args()

    ratio = tuple(args.ratio)
    if abs(sum(ratio) - 1.0) > 1e-6:
        parser.error(f"--ratio values must sum to 1.0, got {sum(ratio):.6f}")

    if args.dst is not None:
        dst = os.path.abspath(args.dst)
    elif args.preprocess:
        dst = DST_INFERENCE_DEFAULT
    else:
        dst = DST_DEFAULT

    for stage in STAGES:
        os.makedirs(os.path.join(dst, stage), exist_ok=True)

    def _resolve(p: str) -> str:
        if os.path.isabs(p):
            return p
        cwd_rel = os.path.abspath(p)
        if os.path.isfile(cwd_rel):
            return cwd_rel
        return str(REPO / p)

    inputs = [_resolve(p) for p in args.inputs]
    missing = [p for p in inputs if not os.path.isfile(p)]
    if missing:
        for p in missing:
            print(f"ERROR: file not found: {p}", file=sys.stderr)
        sys.exit(1)

    # Convert any .txt inputs to .laz before splitting
    laz_inputs = []
    for src_path in inputs:
        if src_path.lower().endswith('.txt'):
            src_path = txt_to_laz(src_path)
        laz_inputs.append(src_path)
    inputs = laz_inputs

    print(f"\nOutput root   : {dst}")
    print(f"Files         : {len(inputs)}")
    print(f"Ratio (T/V/Te): {ratio[0]:.3f} / {ratio[1]:.3f} / {ratio[2]:.3f}")
    print(f"Axis          : {args.axis}")
    print(f"Preprocess    : {'yes — h5 tiles → data/Inference/processed/' if args.preprocess else 'no'}")

    for src_path in inputs:
        process_file(src_path, dst, ratio, args.axis)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n=== Split summary ===")
    for stage in STAGES:
        stage_dir = os.path.join(dst, stage)
        files = sorted(f for f in os.listdir(stage_dir) if f.endswith(".laz"))
        total_mb = sum(
            os.path.getsize(os.path.join(stage_dir, f)) for f in files
        ) / 1e6
        print(f"  {stage:>5}/  {len(files)} file(s)  {total_mb:.0f} MB total")

    # ── Optional SPT preprocessing ───────────────────────────────────────────
    if args.preprocess:
        print("\n=== SPT preprocessing ===")
        _run_preprocessing()
    else:
        print(f"\nNext step: run SPT preprocessing with:")
        print(f"  python scripts/prepare_laz.py <files> --preprocess")
        print(f"  # or manually:")
        print(f"  python src/train.py experiment=semantic/forest trainer.max_epochs=0")


if __name__ == "__main__":
    main()
