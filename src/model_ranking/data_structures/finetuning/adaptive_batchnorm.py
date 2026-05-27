from pydantic import BaseModel
from typing import Optional

from .model import SelfTrainingModelConfig

from model_ranking.data_structures.data import val_loaders_type


class AdaptiveBatchNormConfig(BaseModel):
    model_cfg: SelfTrainingModelConfig
    loaders: val_loaders_type
    output_checkpoint_dir_path: str
    n_patches: Optional[int] = None
    foreground_ratio_threshold: Optional[float] = 0.0

