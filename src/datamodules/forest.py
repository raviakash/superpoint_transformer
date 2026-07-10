import logging
from src.datamodules.base import BaseDataModule
from src.datasets.forest import (
    ForestDataset, MiniForestDataset, InferenceForestDataset, DeadwoodDataset,
)

log = logging.getLogger(__name__)

__all__ = ['ForestDataModule', 'MiniForestDataModule', 'InferenceForestDataModule',
           'DeadwoodDataModule']


class ForestDataModule(BaseDataModule):
    """LightningDataModule for the SegmentedForests dataset."""
    _DATASET_CLASS = ForestDataset
    _MINIDATASET_CLASS = MiniForestDataset


class MiniForestDataModule(ForestDataModule):
    """Mini variant — only the first 2 plots per stage."""
    _DATASET_CLASS = MiniForestDataset


class DeadwoodDataModule(ForestDataModule):
    """DataModule for below-1m HAG crops stored under data/deadwood/."""
    _DATASET_CLASS = DeadwoodDataset


class InferenceForestDataModule(ForestDataModule):
    """DataModule for new unlabelled data stored under data/Inference/.

    Raw LAZ files must be in data/Inference/raw/{train,val,test}/.
    Processed h5 files are written to data/Inference/processed/.
    The existing data/forest/ tree is never touched.
    """
    _DATASET_CLASS = InferenceForestDataset


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = str(pyrootutils.setup_root(__file__, pythonpath=True))
    cfg = omegaconf.OmegaConf.load(root + "/configs/datamodule/semantic/forest.yaml")
    cfg.data_dir = root + "/data"
    _ = hydra.utils.instantiate(cfg)
