from pydantic import BaseModel
from typing import Any, Dict, Optional

from .data import SelfTrainingDataConfig
from .model import SelfTrainingModelConfig
from .pseudo_labeler import pseudo_labeler_type
from .self_training import SelfTrainingTrainConfig

from model_ranking.data_structures.general.logging import WandbConfig


class MeanTeacherConfig(BaseModel):
    name: str
    output_root_path: str
    data_cfg: SelfTrainingDataConfig
    pseudo_labeler_cfg: pseudo_labeler_type
    model_cfg: SelfTrainingModelConfig
    training_cfg: SelfTrainingTrainConfig
    wandb_cfg: Optional[WandbConfig]
    supervised_loader_cfg: Optional[Dict[str, Any]] = None
