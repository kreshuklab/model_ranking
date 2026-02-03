from pydantic import BaseModel
from typing import Any, Dict, Optional

from .model import SelfTrainingModelConfig
from .self_training import SelfTrainingTrainConfig

from model_ranking.data_structures.general.logging import WandbConfig


class SupervisedFinetuningConfig(BaseModel):
    name: str
    output_root_path: str
    model_cfg: SelfTrainingModelConfig
    training_cfg: SelfTrainingTrainConfig
    wandb_cfg: Optional[WandbConfig]
    loader_cfg: Dict[str, Any]
