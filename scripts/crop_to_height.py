"""
Slope-aware crop of a labelled LAZ file to the bottom N metres above terrain.

Uses CSF cloth simulation to estimate the terrain surface, then computes
height-above-ground (HAG) per point via inverse-distance-weighted interpolation
from the cloth nodes.  Points with HAG >= height_cutoff are discarded.

All point attributes (RGB, Class, Split, and any other extra-byte fields) are
preserved.  z coordinates are NOT modified — the output keeps absolute elevation.

Usage
-----
    python scripts/crop_to_height.py plot_01.laz
    python scripts/crop_to_height.py plot_01.laz --height 1.0 --output plot_01_1m.laz

    # Steeper terrain:
    python scripts/crop_to_height.py plot_01.laz --rigidness 1

    # Finer DTM:
    python scripts/crop_to_height.py plot_01.laz --cloth-resolution 0.3
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import laspy
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'scripts'))
from preprocessing_functions import Preprocessing

logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)


def crop_to_height(
    input_path: str,
    output_path: str,
    height_cutoff: float = 1.0,
    cloth_resolution: float = 0.5,
    classify_threshold: float = 0.1,
    rigidness: int = 2,
    bSloopSmooth: bool = True,
    k_neighbours: int = 5,
) -> None:
    log.info("Reading: %s", input_path)
    las = laspy.read(input_path)
    n_before = len(las.points)
    log.info("  Points            : %s", f"{n_before:,}")
    log.info("  Extra-byte fields : %s",
             list(las.point_format.extra_dimension_names) or "none")

    # Build xyz float32 array for CSF — only geometry needed for DTM
    prep = Preprocessing()
    prep.points = np.column_stack([
        np.asarray(las.x, dtype=np.float32),
        np.asarray(las.y, dtype=np.float32),
        np.asarray(las.z, dtype=np.float32),
    ])

    log.info("Fitting DTM via CSF "
             "(cloth_resolution=%.2f m, classify_threshold=%.2f m, rigidness=%d) ...",
             cloth_resolution, classify_threshold, rigidness)
    dtm = prep.generate_dtm(
        bSloopSmooth=bSloopSmooth,
        cloth_resolution=cloth_resolution,
        classify_threshold=classify_threshold,
        rigidness=rigidness,
    )
    log.info("  DTM cloth nodes   : %s", f"{len(dtm):,}")

    log.info("Computing height-above-ground (k_neighbours=%d) ...", k_neighbours)
    normalized = prep.normalize_heights(dtm, k_neighbours=k_neighbours)
    hag = normalized[:, 2]

    mask = hag < height_cutoff
    n_after = int(mask.sum())
    log.info("  HAG range         : %.3f m  to  %.3f m", hag.min(), hag.max())
    log.info("  Kept (HAG < %.1f m): %s  (%.1f%%)",
             height_cutoff, f"{n_after:,}", 100.0 * n_after / n_before)

    # Apply mask — all original attributes and absolute z are preserved
    out_las = laspy.LasData(header=las.header)
    out_las.points = las.points[mask].copy()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    log.info("Writing: %s", output_path)
    out_las.write(output_path)
    log.info("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Slope-aware crop of a labelled LAZ file to the bottom N metres above terrain. "
                    "All attributes (RGB, Class, Split, …) are preserved; z is NOT modified.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input", help="Input .laz or .las file path")
    parser.add_argument(
        "--output", default=None,
        help="Output path (default: <stem>_1m.laz next to the input file)",
    )
    parser.add_argument(
        "--height", type=float, default=1.0,
        help="Height cutoff in metres above terrain (default: 1.0)",
    )
    parser.add_argument(
        "--cloth-resolution", type=float, default=0.5, dest="cloth_resolution",
        help="CSF cloth grid resolution in metres (default: 0.5). "
             "Smaller = finer DTM, slower.",
    )
    parser.add_argument(
        "--classify-threshold", type=float, default=0.1, dest="classify_threshold",
        help="CSF ground/non-ground classification threshold in metres (default: 0.1).",
    )
    parser.add_argument(
        "--rigidness", type=int, default=2, choices=[1, 2, 3],
        help="CSF cloth rigidness: 1=steep terrain, 2=moderate (default), 3=flat.",
    )
    parser.add_argument(
        "--no-slope-smooth", action="store_true",
        help="Disable CSF slope smoothing (enabled by default).",
    )
    parser.add_argument(
        "--k-neighbours", type=int, default=5, dest="k_neighbours",
        help="Number of DTM nodes for IDW interpolation per point (default: 5).",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        log.error("Input file not found: %s", args.input)
        sys.exit(1)

    if args.output is None:
        stem = Path(args.input).stem
        args.output = str(Path(args.input).parent / f"{stem}_1m.laz")

    crop_to_height(
        input_path=args.input,
        output_path=args.output,
        height_cutoff=args.height,
        cloth_resolution=args.cloth_resolution,
        classify_threshold=args.classify_threshold,
        rigidness=args.rigidness,
        bSloopSmooth=not args.no_slope_smooth,
        k_neighbours=args.k_neighbours,
    )


if __name__ == "__main__":
    main()

