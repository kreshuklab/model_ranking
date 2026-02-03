from pydantic import BaseModel
from typing import Sequence, Mapping, Literal, Optional

from .data import augmentation_type


class ClassificationConsistencyMetric(BaseModel):
    name: Literal["EI", "Hamming-Distance"]
    threshold: float
    save_key: str
    overwrite_scores: bool


class ClassificationPredicitonLoadConfig(BaseModel):
    source: Sequence[str]
    target: Sequence[str]
    perturbations: Mapping[augmentation_type, Sequence[str]]
    base_path: str


class ClassificationConsistencyConfig(ClassificationPredicitonLoadConfig):
    consistency_metric: ClassificationConsistencyMetric


class ClassificationSummaryResultsConfig(BaseModel):
    eval_key: Optional[str]
    overwrite_scores: bool
    save_name_postfix: str = ""
