from pydantic import BaseModel, Discriminator
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from model_ranking.data_structures.ranking.consistency_metrics import (
    consistency_metric_type,
)
from model_ranking.data_structures.general.model import Pytorch3DUnetModelConfig
from model_ranking.data_structures.general.segmentation import segmentation_type


class ConsistencyPseudoLabelerConfig(BaseModel):
    # consistency_metric: consistency_metric_type
    consistency_threshold: Optional[float]
    seg_params: segmentation_type
    consistency_metric: consistency_metric_type
    activation: Optional[Literal["softmax", "sigmoid"]] = "sigmoid"


class InputConsisPseudoLabelerConfig(ConsistencyPseudoLabelerConfig):
    name: Literal["input_consistency"]
    transformer_cfg: Dict[str, List[Any]]
    stats_cfg: Dict[str, Any]


class ModelConsisPseudoLabelerConfig(ConsistencyPseudoLabelerConfig):
    name: Literal["model_consistency"]
    perturbed_model_config: Pytorch3DUnetModelConfig


class DefaultPseudoLabelerConfig(BaseModel):
    name: Literal["default_pseudo_labeler"]
    confidence_threshold: Optional[float] = None
    threshold_from_both_sides: bool = True
    mask_channel: Optional[int] = None
    activation: Optional[Literal["softmax", "sigmoid"]] = "sigmoid"


class DummyDirectEvalPseudoLabelerConfig(BaseModel):
    name: Literal["direct_eval_pseudo_labeler"]
    score_threshold: float = 0.5
    activation: Optional[Literal["softmax", "sigmoid"]] = "sigmoid"


class ScheduledPseudoLabelerConfig(BaseModel):
    name: Literal["scheduled_pseudo_labeler"]
    confidence_threshold: Optional[float] = None
    threshold_from_both_sides: bool = True
    mode: Literal["min", "max"] = "min"
    factor: float = 0.05
    patience: int = 10
    threshold: float = 1e-4
    threshold_mode: Literal["rel", "abs"] = "abs"
    min_ct: float = 0.5
    eps: float = 1e-8
    verbose: bool = True
    activation: Optional[Literal["softmax", "sigmoid"]] = "sigmoid"


psuedo_labeler_names = Literal[
    "input_consistency",
    "model_consistency",
    "default_pseudo_labeler",
    "scheduled_pseudo_labeler",
    "direct_eval_pseudo_labeler",
]


pseudo_labeler_type = Annotated[
    Union[
        InputConsisPseudoLabelerConfig,
        ModelConsisPseudoLabelerConfig,
        DefaultPseudoLabelerConfig,
        ScheduledPseudoLabelerConfig,
        DummyDirectEvalPseudoLabelerConfig,
    ],
    Discriminator("name"),
]
