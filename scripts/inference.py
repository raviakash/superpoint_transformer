"""
Forest dataset inference script.

Loads a trained SPT checkpoint, runs the predict loop on the test (or val)
split, and for each plot:
  - back-projects voxel predictions to the full-resolution raw LAZ
  - writes an output LAZ with a PredictedClass extra-byte field and
    prediction-coloured RGB (CloudCompare / potree ready)
  - prints per-class IoU when ground-truth labels are present

Usage
-----
    # Direct inference on a new .txt or .laz file (no manual prep needed):
    python scripts/inference.py --ckpt <ckpt> --input /path/to/cloud.txt
    python scripts/inference.py --ckpt <ckpt> --input /path/to/cloud.laz

    # Existing split-based flow:
    python scripts/inference.py \\
        --ckpt logs/train/runs/2026-06-11_15-11-18/checkpoints/epoch_379.ckpt

    python scripts/inference.py --ckpt <ckpt> --split val
    python scripts/inference.py --ckpt <ckpt> --out results/my_run/
"""

import os
import sys
import shutil
import logging
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import laspy
from pytorch_lightning import Callback, Trainer

# ── repo root on sys.path ─────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.datasets.forest_config import (
    CLASS_NAMES, CLASS_COLORS, FOREST_NUM_CLASSES, ID2TRAINID)

log = logging.getLogger(__name__)
logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)

# ── train ID → original dataset class ID ─────────────────────────────────────
_TRAINID2RAWID = np.full(FOREST_NUM_CLASSES + 1, 255, dtype=np.uint8)
for _raw_id, _tid in enumerate(ID2TRAINID):
    if _tid <= FOREST_NUM_CLASSES and _TRAINID2RAWID[_tid] == 255:
        _TRAINID2RAWID[_tid] = _raw_id


# ─────────────────────────────────────────────────────────────────────────────
# Callback: collect per-voxel predictions from predict_step
# ─────────────────────────────────────────────────────────────────────────────

class ForestPredictionCallback(Callback):
    """
    Hooks into on_predict_batch_end to collect (voxel_pos, voxel_pred) per
    tile and group them by raw LAZ file.

    predict_step() returns (nag, output):
      - nag[0].pos            float32  [N_vox, 3]  local coords
      - nag[0].super_index    long     [N_vox]      maps voxels → superpoints
      - output.semantic_pred()  long   [N_sp]       per-superpoint argmax

    All tiles from the same raw plot share the same local coordinate origin
    (the position of raw_pos[0] in read_forest_plot). This is guaranteed
    because tiling happens AFTER reading the full file in BaseDataset.process().

    :param dataset: the ForestDataset for the target split (test or val)
    """

    def __init__(self, dataset):
        # plot_data[raw_rel_path] = {'pos': [arr, ...], 'pred': [arr, ...]}
        self.dataset = dataset
        self.plot_data: dict = defaultdict(lambda: {"pos": [], "pred": []})
        self._item_count = 0   # running total of items seen (not batches)

    def on_predict_start(self, trainer, pl_module):
        self._item_count = 0

    def on_predict_batch_end(
            self, trainer, pl_module, outputs, batch,
            batch_idx, dataloader_idx=0):
        nag, output = outputs

        # Actual number of samples in this batch (may be < batch_size at the end)
        n_in_batch = int(nag[0].batch.max().item()) + 1 \
            if (hasattr(nag[0], "batch") and nag[0].batch is not None) \
            else 1

        # id_to_relative_raw_path strips the tile suffix internally.
        # Use _item_count (item index) not batch_idx (batch index).
        cloud_id = self.dataset.cloud_ids[self._item_count]
        raw_rel  = self.dataset.id_to_relative_raw_path(cloud_id)

        # Per-voxel predictions via the super_index broadcast.
        sp_pred     = output.semantic_pred().detach().cpu()      # [N_sp total]
        super_index = nag[0].super_index.detach().cpu()          # [N_vox total]
        vox_pred    = sp_pred[super_index].numpy().astype(np.int64)
        vox_pos     = nag[0].pos.detach().cpu().numpy().astype(np.float32)

        self.plot_data[raw_rel]["pos"].append(vox_pos)
        self.plot_data[raw_rel]["pred"].append(vox_pred)

        self._item_count += n_in_batch


# ─────────────────────────────────────────────────────────────────────────────
# Back-projection: voxel predictions → raw LAZ points (1-NN)
# ─────────────────────────────────────────────────────────────────────────────

def backproject(vox_pos: np.ndarray, vox_pred: np.ndarray,
                local_raw_pos: np.ndarray) -> np.ndarray:
    """
    For each raw point assign the class of its nearest voxel (1-NN).

    Uses scipy.cKDTree — O(N log N) time, O(N) memory; handles millions
    of points without CUDA memory issues.

    Both vox_pos and local_raw_pos must be in the same local coordinate
    space (relative to raw_pos[0] of the source LAZ).
    """
    from scipy.spatial import cKDTree
    tree = cKDTree(vox_pos)
    _, idx = tree.query(local_raw_pos, k=1, workers=-1)
    return vox_pred[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Write output LAZ
# ─────────────────────────────────────────────────────────────────────────────

def write_laz(src_path: str, pred: np.ndarray, out_path: str) -> None:
    """
    Copy the raw LAZ, add PredictedClass / PredictedRawClass extra-byte
    fields, and overwrite RGB with per-class colours for easy visualisation.
    """
    las = laspy.read(src_path)
    assert len(pred) == len(las.points), (
        f"Prediction length {len(pred)} != point count {len(las.points)}")

    pred_u8  = pred.clip(0, FOREST_NUM_CLASSES).astype(np.uint8)
    raw_cls  = _TRAINID2RAWID[pred_u8]
    # CLASS_COLORS has 16 rows; void (16) → black
    colours  = np.vstack([CLASS_COLORS, [[0, 0, 0]]])[pred_u8]

    las.add_extra_dims([
        laspy.ExtraBytesParams(
            "PredictedClass",    type=np.uint8,
            description="SPT train class ID 0-15"),
        laspy.ExtraBytesParams(
            "PredictedRawClass", type=np.uint8,
            description="Original dataset class ID"),
    ])
    las["PredictedClass"]    = pred_u8
    las["PredictedRawClass"] = raw_cls

    # LAS RGB is uint16; shift uint8 colour to the high byte
    las.red   = colours[:, 0].astype(np.uint16) << 8
    las.green = colours[:, 1].astype(np.uint16) << 8
    las.blue  = colours[:, 2].astype(np.uint16) << 8

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    las.write(out_path)


# ─────────────────────────────────────────────────────────────────────────────
# IoU table
# ─────────────────────────────────────────────────────────────────────────────

def print_iou_table(gt: np.ndarray, pred: np.ndarray) -> None:
    iou = np.full(FOREST_NUM_CLASSES, np.nan)
    for c in range(FOREST_NUM_CLASSES):
        tp = int(((gt == c) & (pred == c)).sum())
        fp = int(((gt != c) & (pred == c)).sum())
        fn = int(((gt == c) & (pred != c)).sum())
        if (tp + fp + fn) > 0:
            iou[c] = tp / (tp + fp + fn)

    miou = float(np.nanmean(iou))

    print("\n" + "═" * 62)
    print("  TEST RESULTS  (point-level IoU)")
    print("═" * 62)
    print(f"  {'Class':<28}  {'IoU':>6}   Bar")
    print("─" * 62)
    for c, name in enumerate(CLASS_NAMES[:FOREST_NUM_CLASSES]):
        v = iou[c]
        if np.isnan(v):
            bar, pct = "░" * 20, "  N/A"
        else:
            bar = "█" * int(v * 20) + "░" * (20 - int(v * 20))
            pct = f"{v * 100:5.1f}%"
        print(f"  {name:<28}  {pct:>6}   {bar}")
    print("─" * 62)
    print(f"  {'mIoU':<28}  {miou * 100:5.1f}%")
    print("═" * 62)


# ─────────────────────────────────────────────────────────────────────────────
# Direct-input helper: convert + stage a single file for inference
# ─────────────────────────────────────────────────────────────────────────────

def _prepare_inference_input(input_path: str, data_dir: str) -> None:
    """Convert .txt → .laz if needed, then place in <data_dir>/raw/test/.

    Clears any other .laz files from raw/test/ and removes the processed h5
    cache when the file stem changes, so the model always sees exactly one
    cloud in the test split.
    """
    input_path = os.path.abspath(input_path)

    # ── Convert .txt → .laz alongside the source file ────────────────────────
    if input_path.lower().endswith('.txt'):
        log.info(f"Converting .txt → .laz …")
        sys.path.insert(0, str(REPO / 'scripts'))
        from prepare_laz import txt_to_laz
        laz_path = txt_to_laz(input_path)
    else:
        laz_path = input_path

    stem     = Path(laz_path).stem
    test_dir = os.path.join(data_dir, 'raw', 'test')
    os.makedirs(test_dir, exist_ok=True)
    dst      = os.path.join(test_dir, f'{stem}.laz')

    # ── Detect if the test content is changing ────────────────────────────────
    existing = [f for f in os.listdir(test_dir) if f.lower().endswith('.laz')]
    existing_stems = {os.path.splitext(f)[0] for f in existing}
    stem_changed = existing_stems and existing_stems != {stem}

    if stem_changed:
        # Remove stale processed h5 files so the new cloud gets preprocessed
        proc_dir = os.path.join(data_dir, 'processed')
        if os.path.isdir(proc_dir):
            shutil.rmtree(proc_dir)
            log.info("Cleared old processed h5 (different input — will reprocess)")
        # Remove stale LAZ files
        for f in existing:
            os.remove(os.path.join(test_dir, f))

    # ── Place LAZ in raw/test/ (skip copy if already there) ──────────────────
    already_there = os.path.isfile(dst) and os.path.samefile(laz_path, dst)
    if not already_there:
        shutil.copy2(laz_path, dst)
        log.info(f"Staged → {dst}")
    else:
        log.info(f"Already in place: {dst}")


# ─────────────────────────────────────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_cfg(ckpt_path: str, split: str = "test",
              datamodule_target: str = None,
              experiment: str = "semantic/forest"):
    import pyrootutils
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    # Sets PROJECT_ROOT env var required by configs/paths/default.yaml
    pyrootutils.setup_root(str(REPO), indicator=".project-root", pythonpath=True)

    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)

    overrides = [
        f"experiment={experiment}",
        "trainer=gpu",
        f"ckpt_path={ckpt_path}",
        "logger=csv",
        "extras.print_config=false",
    ]
    if datamodule_target:
        overrides.append(f"datamodule._target_={datamodule_target}")
    # For test split we only need the test dataset; skip train+val setup.
    # For val split we need val_dataset, so we let the full setup run.
    if split == "test":
        overrides.append("datamodule.prepare_only_test=true")

    with initialize_config_dir(
            config_dir=str(REPO / "configs"), version_base="1.2"):
        cfg = compose(config_name="eval", overrides=overrides)
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

# Map data-dir basename → datamodule _target_ class path
_DATADIR_TO_TARGET = {
    'forest':    'src.datamodules.forest.ForestDataModule',
    'Inference': 'src.datamodules.forest.InferenceForestDataModule',
}


def main():
    parser = argparse.ArgumentParser(description="SPT Forest inference")
    parser.add_argument("--ckpt",  required=True,
                        help="Path to trained .ckpt file")
    parser.add_argument("--input", default=None, metavar="FILE",
                        help="Direct inference on a single .txt or .laz file.  "
                             "Automatically converts, stages, preprocesses, and "
                             "runs inference.  Implies --data-dir data/Inference "
                             "and --split test.")
    parser.add_argument("--split", default="test", choices=["test", "val", "train"],
                        help="Data split to evaluate (default: test)")
    parser.add_argument("--data-dir", default=None, dest="data_dir",
                        help="Root dataset directory, e.g. data/forest (default) "
                             "or data/Inference.  Controls which raw LAZ files are "
                             "read and where predicted LAZ files are written.")
    parser.add_argument("--out",   default=None,
                        help="Output directory for predicted LAZ files "
                             "(default: <data-dir>/predictions/<split>/)")
    parser.add_argument("--experiment", default="semantic/forest",
                        help="Hydra experiment config to use for preprocessing "
                             "(default: semantic/forest). Use semantic/forest_deadwood "
                             "when running inference with a deadwood checkpoint.")
    args = parser.parse_args()

    def _resolve(p: str) -> str:
        if os.path.isabs(p):
            return p
        cwd_rel = os.path.abspath(p)
        if os.path.exists(cwd_rel):
            return cwd_rel
        return str(REPO / p)

    ckpt_path = _resolve(args.ckpt)
    if not os.path.exists(ckpt_path):
        log.error(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    # ── --input: convert + stage the file, then run on test split ────────────
    if args.input:
        input_path = _resolve(args.input)
        if not os.path.isfile(input_path):
            log.error(f"Input file not found: {input_path}")
            sys.exit(1)
        data_dir = str(REPO / "data" / "Inference")
        args.split = "test"
        _prepare_inference_input(input_path, data_dir)
    else:
        data_dir = _resolve(args.data_dir) if args.data_dir \
                   else str(REPO / "data" / "forest")

    subdir = Path(data_dir).name
    datamodule_target = _DATADIR_TO_TARGET.get(subdir)
    if datamodule_target is None:
        log.error(
            f"Unknown data-dir '{subdir}'. "
            f"Supported: {list(_DATADIR_TO_TARGET.keys())}")
        sys.exit(1)

    raw_base = os.path.join(data_dir, "raw")
    out_dir  = args.out or os.path.join(data_dir, "predictions", args.split)

    log.info(f"Checkpoint : {ckpt_path}")
    log.info(f"Data dir   : {data_dir}")
    log.info(f"Split      : {args.split}")
    log.info(f"Output dir : {out_dir}")

    # ── 1. Hydra config + instantiation ──────────────────────────────────────
    cfg = build_cfg(ckpt_path, args.split, datamodule_target=datamodule_target,
                    experiment=args.experiment)
    import hydra.utils as hu

    datamodule = hu.instantiate(cfg.datamodule)
    model      = hu.instantiate(cfg.model)

    # ── 2. Setup datamodule and select the right split ────────────────────────
    # For inference we ALWAYS need full-tile batches (no subgraph sampling).
    # predict_dataloader() returns full-tile batches from test_dataset, so we
    # point test_dataset at whichever split we want and use that path for all
    # three splits.  The training/val dataloaders use subgraph sampling which
    # triggers step_multi_run_inference and fails without transform metadata.
    if args.split == "test":
        datamodule.setup("predict")
        active_dataset = datamodule.test_dataset
    elif args.split == "val":
        datamodule.setup("fit")
        active_dataset = datamodule.val_dataset
        datamodule.test_dataset = datamodule.val_dataset
    else:  # train
        datamodule.setup("fit")
        active_dataset = datamodule.train_dataset
        datamodule.test_dataset = datamodule.train_dataset

    predict_kwargs = dict(datamodule=datamodule)

    n_tiles = len(active_dataset.cloud_ids)
    n_plots = len({active_dataset.id_to_relative_raw_path(c) for c in active_dataset.cloud_ids})
    log.info(f"Tiles      : {n_tiles}  ({n_plots} plots × ~{n_tiles // n_plots} tiles each)")

    # ── 3. Prediction loop ────────────────────────────────────────────────────
    collector = ForestPredictionCallback(active_dataset)

    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        logger=False,
        enable_progress_bar=True,
        callbacks=[collector],
    )

    # return_predictions=False avoids accumulating (nag, output) in RAM.
    trainer.predict(
        model=model,
        ckpt_path=ckpt_path,
        return_predictions=False,
        **predict_kwargs,
    )

    log.info(f"\nPredictions collected for {len(collector.plot_data)} plots.")

    # ── 4. Per-plot back-projection and output ────────────────────────────────
    all_gt, all_pred = [], []

    for raw_rel, tile_data in sorted(collector.plot_data.items()):
        raw_path = os.path.join(raw_base, raw_rel)
        plot_name = os.path.splitext(os.path.basename(raw_path))[0]
        log.info(f"\n  {plot_name}")

        if not os.path.exists(raw_path):
            log.warning(f"    Raw LAZ not found: {raw_path} — skipping")
            continue

        # Load raw LAZ
        las = laspy.read(raw_path)
        raw_pos = np.stack([
            np.asarray(las.x, dtype=np.float64),
            np.asarray(las.y, dtype=np.float64),
            np.asarray(las.z, dtype=np.float64),
        ], axis=-1)

        # Convert to local coords matching the NAG (origin = raw_pos[0]).
        offset        = raw_pos[0].copy()
        local_raw_pos = (raw_pos - offset).astype(np.float32)

        # Accumulate voxel predictions from all tiles belonging to this plot.
        vox_pos  = np.concatenate(tile_data["pos"],  axis=0)
        vox_pred = np.concatenate(tile_data["pred"], axis=0)
        log.info(f"    Raw pts: {len(raw_pos):,}   Voxels: {len(vox_pos):,}")

        # 1-NN back-projection in local coordinate space
        pred_bp = backproject(vox_pos, vox_pred, local_raw_pos)

        # Collect GT labels if available
        try:
            raw_cls  = np.asarray(las["Class"], dtype=np.int64)
            gt_train = ID2TRAINID[raw_cls.clip(0, len(ID2TRAINID) - 1)]
            all_gt.append(gt_train)
            all_pred.append(pred_bp)
        except Exception:
            pass

        # Write predicted LAZ
        out_path = os.path.join(out_dir, f"{plot_name}_pred.laz")
        write_laz(raw_path, pred_bp, out_path)
        log.info(f"    Saved → {out_path}")

    # ── 5. IoU table ─────────────────────────────────────────────────────────
    if all_gt:
        gt_arr   = np.concatenate(all_gt)
        pred_arr = np.concatenate(all_pred)
        valid    = gt_arr < FOREST_NUM_CLASSES
        print_iou_table(gt_arr[valid], pred_arr[valid])
    else:
        log.info("\nNo ground-truth labels found — skipping IoU table.")

    log.info(f"\nDone.  Output files: {out_dir}")


if __name__ == "__main__":
    main()
