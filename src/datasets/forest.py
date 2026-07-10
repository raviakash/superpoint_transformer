import os
import sys
import torch
import logging
import os.path as osp
from typing import List

import laspy
import numpy as np

from src.datasets import BaseDataset
from src.data import Data
from src.datasets.forest_config import (
    CLASS_NAMES, CLASS_COLORS, FOREST_NUM_CLASSES,
    STUFF_CLASSES, TILES, ID2TRAINID,
    TRAIN_RGB_MEAN, TRAIN_RGB_STD,
)

DIR = os.path.dirname(os.path.realpath(__file__))
log = logging.getLogger(__name__)

# Occasional DataLoader issues with large point clouds on some machines.
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

__all__ = ['ForestDataset', 'MiniForestDataset']


########################################################################
#                               Utils                                  #
########################################################################

def read_forest_plot(
        filepath: str,
        xyz: bool = True,
        rgb: bool = True,
        semantic: bool = True,
        remap: bool = True,
        normalize_rgb: bool = False,
) -> Data:
    """Read a single pre-split forest LAZ file and return a Data object.

    :param filepath: absolute path to the .laz file
    :param xyz:      store XYZ in Data.pos (with float64 offset in Data.pos_offset)
    :param rgb:      store normalised RGB in Data.rgb  (float32, range [0, 1])
    :param semantic: store class labels in Data.y
    :param remap:    remap raw class IDs (0-13, 22, 23) → train IDs (0-15);
                     anything outside the mapping is set to 16 (void)
    """
    data = Data()
    las = laspy.read(filepath)

    if xyz:
        pos = torch.stack([
            torch.tensor(np.array(las.x), dtype=torch.float),
            torch.tensor(np.array(las.y), dtype=torch.float),
            torch.tensor(np.array(las.z), dtype=torch.float),
        ], dim=-1)
        # Subtract the first point so coordinates stay in a small float32 range.
        # The full-precision offset is stored separately for reconstruction.
        pos_offset = pos[0].clone().to(torch.double)
        data.pos = pos - pos_offset.float()
        data.pos_offset = pos_offset

    if rgb:
        # LAS stores RGB as uint16 (0–65535); cast via float32 then normalise to [0, 1].
        # Guard against LAZ files with no RGB channels (unlabelled/sensor-only data).
        try:
            r = np.array(las.red,   dtype=np.float32)
            g = np.array(las.green, dtype=np.float32)
            b = np.array(las.blue,  dtype=np.float32)
        except Exception:
            n = len(las.points)
            r = g = b = np.zeros(n, dtype=np.float32)

        if normalize_rgb:
            # Z-score normalise each channel from the file's own distribution to
            # the training data distribution, so Cut-Pursuit and the neural network
            # see in-distribution colour features regardless of sensor radiometry.
            for ch, tgt_mean, tgt_std in zip(
                    [r, g, b], TRAIN_RGB_MEAN, TRAIN_RGB_STD):
                src_mean = float(ch.mean())
                src_std  = float(ch.std())
                if src_std < 1.0:
                    src_std = 1.0
                ch[:] = np.clip(
                    (ch - src_mean) / src_std * tgt_std + tgt_mean,
                    0.0, 65535.0)

        data.rgb = torch.stack([
            torch.from_numpy(r) / 65535.0,
            torch.from_numpy(g) / 65535.0,
            torch.from_numpy(b) / 65535.0,
        ], dim=-1)

    if semantic:
        _extra_names = set(las.point_format.extra_dimension_names)
        if 'Class' in _extra_names:
            raw_y = torch.tensor(np.array(las['Class']), dtype=torch.long)
        else:
            # No Class field — treat all points as void so they're excluded from loss/IoU
            raw_y = torch.full((len(las.points),), FOREST_NUM_CLASSES, dtype=torch.long)
        if remap:
            # Clamp to valid index range before lookup to avoid index errors
            safe_y = raw_y.clamp(0, len(ID2TRAINID) - 1)
            data.y = torch.from_numpy(ID2TRAINID)[safe_y]
        else:
            data.y = raw_y

    return data


########################################################################
#                          ForestDataset                               #
########################################################################

class ForestDataset(BaseDataset):
    """SegmentedForests dataset — 14 TLS/MLS forest plots, 16 classes.

    Dataset: https://zenodo.org/records/17396681
    Paper:   Forestry: An International Journal of Forest Research (2025),
             doi:10.1093/forestry/cpaf062

    Before instantiating this dataset for the first time, run:
        python scripts/prepare_forest_data.py
    to split the original LAZ files into train/val/test sub-directories.
    """

    @property
    def class_names(self) -> List[str]:
        return CLASS_NAMES

    @property
    def num_classes(self) -> int:
        return FOREST_NUM_CLASSES

    @property
    def stuff_classes(self) -> List[int]:
        return STUFF_CLASSES

    @property
    def class_colors(self):
        return CLASS_COLORS

    @property
    def data_subdir_name(self) -> str:
        return 'forest'

    @property
    def all_base_cloud_ids(self):
        """Scan raw/{train,val,test}/ for .laz files and build cloud ID lists.

        Falls back to the hardcoded TILES constant when the raw directory does
        not yet exist (e.g. before prepare_forest_data.py has been run).

        Convention (matches id_to_relative_raw_path):
          train/foo.laz  →  cloud_id 'foo'
          val/foo.laz    →  cloud_id 'foo_val'
          test/foo.laz   →  cloud_id 'foo_test'

        NOTE: BaseDataset calls check_cloud_ids() BEFORE super().__init__() sets
        self.root, so self.raw_dir is unavailable at that point. Fall back to TILES
        so the uniqueness check passes; real IDs are resolved on every later call.
        """
        try:
            raw_dir = self.raw_dir
        except AttributeError:
            return TILES
        result = {}
        has_any = False
        for stage, suffix in (('train', ''), ('val', '_val'), ('test', '_test')):
            stage_dir = osp.join(raw_dir, stage)
            if osp.isdir(stage_dir):
                ids = sorted(
                    osp.splitext(f)[0] + suffix
                    for f in os.listdir(stage_dir)
                    if f.lower().endswith('.laz')
                )
                result[stage] = ids
                if ids:
                    has_any = True
            else:
                result[stage] = []
        return result if has_any else TILES

    def download_dataset(self) -> None:
        log.error(
            "\nForestDataset: raw LAZ files not found.\n"
            "1. Download SegmentedForests.zip from https://zenodo.org/records/17396681\n"
            f"   and extract plot_01.laz … plot_14.laz to:  {self.root}/\n"
            "2. Run the pre-split script:\n"
            "   python scripts/prepare_forest_data.py\n"
            f"   Expected output layout:\n{self.raw_file_structure}\n"
        )
        sys.exit(1)

    def read_single_raw_cloud(self, raw_cloud_path: str) -> Data:
        return read_forest_plot(
            raw_cloud_path, xyz=True, rgb=True, semantic=True, remap=True)

    @property
    def raw_file_structure(self) -> str:
        return (
            f"\n    {self.root}/\n"
            "        └── raw/\n"
            "            ├── train/\n"
            "            │   └── plot_{01,...,14}.laz\n"
            "            ├── val/\n"
            "            │   └── plot_{01,...,14}.laz\n"
            "            └── test/\n"
            "                └── plot_{01,...,14}.laz\n"
        )

    def id_to_relative_raw_path(self, id: str) -> str:
        """Map a cloud ID to its path relative to self.raw_dir.

        Cloud ID conventions:
          'plot_01'       → train/plot_01.laz
          'plot_01_val'   → val/plot_01.laz
          'plot_01_test'  → test/plot_01.laz
        Tiling suffixes (e.g. __TILE_1_OF_4) are stripped first by
        id_to_base_id() before this method is called.
        """
        base_id = self.id_to_base_id(id)
        if base_id.endswith('_test'):
            return osp.join('test', base_id[:-5] + '.laz')
        if base_id.endswith('_val'):
            return osp.join('val', base_id[:-4] + '.laz')
        return osp.join('train', base_id + '.laz')

    def processed_to_raw_path(self, processed_path: str) -> str:
        """Map a processed .h5 path back to the corresponding raw LAZ."""
        _, _, cloud_id = osp.splitext(processed_path)[0].split(os.sep)[-3:]
        return osp.join(self.raw_dir, self.id_to_relative_raw_path(cloud_id))


########################################################################
#                       InferenceForestDataset                         #
########################################################################

class InferenceForestDataset(ForestDataset):
    """ForestDataset rooted at data/Inference/ for new, unseen point clouds.

    All raw LAZ files are read from data/Inference/raw/{train,val,test}/ and
    processed h5 files are written to data/Inference/processed/.  The existing
    data/forest/ directory is left completely untouched.

    RGB channels are z-score normalised to match the training data distribution
    before voxelisation, so Cut-Pursuit and the neural network receive
    in-distribution colour features even when the sensor radiometry differs.
    """

    @property
    def data_subdir_name(self) -> str:
        return 'Inference'

    def read_single_raw_cloud(self, raw_cloud_path: str) -> Data:
        return read_forest_plot(
            raw_cloud_path, xyz=True, rgb=True, semantic=True, remap=True,
            normalize_rgb=True)


########################################################################
#                        DeadwoodDataset                               #
########################################################################

class DeadwoodDataset(ForestDataset):
    """ForestDataset rooted at data/deadwood/ for below-1m HAG crops.

    Place pre-cropped, height-normalised LAZ files (with Class extra-byte)
    in data/deadwood/raw/{train,val,test}/ before running preprocessing.
    Class labels must already be train IDs (0-15) — no ID2TRAINID remapping
    is applied (remap=False).
    """

    @property
    def data_subdir_name(self) -> str:
        return 'deadwood'

    def read_single_raw_cloud(self, raw_cloud_path: str) -> Data:
        return read_forest_plot(
            raw_cloud_path, xyz=True, rgb=True, semantic=True, remap=False)


########################################################################
#                         MiniForestDataset                            #
########################################################################

class MiniForestDataset(ForestDataset):
    """Mini version: first 2 plots per stage — fast experimentation."""

    _NUM_MINI = 2

    @property
    def all_base_cloud_ids(self):
        return {k: v[:self._NUM_MINI] for k, v in super().all_base_cloud_ids.items()}

    @property
    def data_subdir_name(self) -> str:
        return 'forest'

    def process(self):
        super().process()

    def download(self):
        super().download()
