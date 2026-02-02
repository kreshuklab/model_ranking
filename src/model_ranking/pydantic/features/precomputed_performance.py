from pydantic import BaseModel
from typing import Literal, Union


class PrecomputedPerformanceConfig(BaseModel):
    base_path: str
    key: str
    invert_score: bool = False
    metric_summary: bool = False


class PrecomputedDirectPerformanceConfig(PrecomputedPerformanceConfig):
    name: Literal["direct_performance"]
    approach: Literal["consistency", "feature_perturbation_consistency"]
    run_id: str


class PrecomputedFinetunedPerformanceConfig(PrecomputedPerformanceConfig):
    name: Literal["finetuned_performance"]
    finetuning_approach: Literal[
        "confidence_threshold",
        "direct_eval",
        "feature_perturbation",
        "default_selftraining",
        "AdaBN",
    ]
    result_type: Literal["predictions", "checkpoints"]


class PrecomputedClassificationPerformanceConfig(PrecomputedPerformanceConfig):
    name: Literal["classification_performance"]


performance_type = Union[
    PrecomputedDirectPerformanceConfig,
    PrecomputedFinetunedPerformanceConfig,
    PrecomputedClassificationPerformanceConfig,
]

segmentation_performance_type = Union[
    PrecomputedDirectPerformanceConfig,
    PrecomputedFinetunedPerformanceConfig,
]
