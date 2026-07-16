"""
Height-normalize a labelled LAZ file using CSF cloth simulation.

All point attributes (RGB, Class, Split, and any other extra-byte fields)
are preserved unchanged.  Only the z coordinate is replaced with
height-above-ground (HAG) values computed by:
  1. Fitting a Digital Terrain Model (DTM) with the Cloth Simulation Filter
  2. Subtracting the IDW-interpolated DTM elevation from each point's z

Usage
-----
    python scripts/height_normalize.py plot_01.laz
    python scripts/height_normalize.py plot_01.laz --output plot_01_hag.laz

    # Steeper terrain — use more flexible cloth:
    python scripts/height_normalize.py plot_01.laz --rigidness 1

    # Finer DTM for complex micro-topography:
    python scripts/height_normalize.py plot_01.laz --cloth-resolution 0.3
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


def height_normalize_laz(
    input_path: str,
    output_path: str,
    cloth_resolution: float = 0.5,
    classify_threshold: float = 0.1,
    rigidness: int = 2,
    bSloopSmooth: bool = True,
    k_neighbours: int = 5,
) -> None:
    log.info("Reading: %s", input_path)
    las = laspy.read(input_path)
    log.info("  Points         : %s", f"{len(las.points):,}")
    log.info("  Extra-byte fields: %s",
             list(las.point_format.extra_dimension_names) or "none")

    # Build xyz float32 array for CSF — only geometry is needed for DTM
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
    log.info("  DTM cloth nodes: %s", f"{len(dtm):,}")

    log.info("Normalizing heights (k_neighbours=%d) ...", k_neighbours)
    normalized = prep.normalize_heights(dtm, k_neighbours=k_neighbours)
    hag = normalized[:, 2].astype(np.float64)

    log.info("  HAG range: %.3f m  to  %.3f m", hag.min(), hag.max())

    # Copy the full LAZ (all points, all extra bytes, same header), replace z only.
    # laspy stores x/y/z as scaled integers named 'X','Y','Z' (uppercase) in the
    # raw numpy point array.  Setting out_las.z on a copied LasData fails in some
    # laspy versions, so write the scaled integer directly.
    out_las = laspy.LasData(header=las.header)
    out_las.points = las.points.copy()
    scale_z  = float(las.header.scales[2])
    offset_z = float(las.header.offsets[2])
    out_las.points.array['Z'] = np.round((hag - offset_z) / scale_z).astype(np.int32)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    log.info("Writing: %s", output_path)
    out_las.write(output_path)
    log.info("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Height-normalize a labelled LAZ file. "
                    "All attributes (RGB, Class, Split, …) are preserved; "
                    "only z is replaced with height-above-ground.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input", help="Input .laz or .las file path")
    parser.add_argument(
        "--output", default=None,
        help="Output path (default: <stem>_hag.laz in the same directory as input)",
    )
    parser.add_argument(
        "--cloth-resolution", type=float, default=0.5, dest="cloth_resolution",
        help="CSF cloth grid resolution in metres (default: 0.5). "
             "Smaller values give a finer DTM but are slower.",
    )
    parser.add_argument(
        "--classify-threshold", type=float, default=0.1, dest="classify_threshold",
        help="CSF ground/non-ground classification threshold in metres (default: 0.1).",
    )
    parser.add_argument(
        "--rigidness", type=int, default=2, choices=[1, 2, 3],
        help="CSF cloth rigidness: 1=most flexible (steep terrain), "
             "2=moderate (default), 3=most rigid (flat terrain).",
    )
    parser.add_argument(
        "--no-slope-smooth", action="store_true",
        help="Disable CSF slope smoothing (enabled by default).",
    )
    parser.add_argument(
        "--k-neighbours", type=int, default=5, dest="k_neighbours",
        help="Number of DTM cloth nodes for IDW height interpolation (default: 5).",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        log.error("Input file not found: %s", args.input)
        sys.exit(1)

    if args.output is None:
        stem = Path(args.input).stem
        args.output = str(Path(args.input).parent / f"{stem}_hag.laz")

    height_normalize_laz(
        input_path=args.input,
        output_path=args.output,
        cloth_resolution=args.cloth_resolution,
        classify_threshold=args.classify_threshold,
        rigidness=args.rigidness,
        bSloopSmooth=not args.no_slope_smooth,
        k_neighbours=args.k_neighbours,
    )


if __name__ == "__main__":
    main()
