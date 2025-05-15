from model_ranking import HammingDistanceEval
from pytorch3dunet.datasets.hdf5 import StandardHDF5Dataset
from pytorch3dunet.augment.transforms import (
    Transformer,
)

pred_loader_cfg = pred_loader.create_config(
    output_dir=pred_dir_path,
    data_base_path=meta_cfg.data_base_path,
)
