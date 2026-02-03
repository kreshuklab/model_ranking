from pydantic import BaseModel
from .model import SelfTrainingModelConfig

from model_ranking.pydantic.data import test_loaders_type


class AdaptiveBatchNormConfig(BaseModel):
    model_cfg: SelfTrainingModelConfig
    loaders: test_loaders_type
    output_checkpoint_dir_path: str
