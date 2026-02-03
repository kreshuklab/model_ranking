from typing import Union
from pydantic import BaseModel

from .data.dataloaders import loader_type
from .general.logging import WandbConfig
from .general.model import Pytorch3DUnetModelConfig
from .general.predictor import predictor_semantic_type, predictor_instance_type
from .general.summary import SummaryResultsConfig

from .meta_config import EvaluateConfig, ConsistencyConfig


class ConfigFull(BaseModel):
    wandb: WandbConfig
    model_path: str
    summary_results: SummaryResultsConfig
    model: Pytorch3DUnetModelConfig
    predictor: Union[predictor_semantic_type, predictor_instance_type]
    loaders: loader_type
    evaluation: EvaluateConfig
    consistency: ConsistencyConfig


class ConfigEvaluation(BaseModel):
    save_results: SummaryResultsConfig
    evaluation: EvaluateConfig


class ConfigConsistency(BaseModel):
    save_results: SummaryResultsConfig
    consistency: ConsistencyConfig
