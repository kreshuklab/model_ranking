from pydantic import BaseModel, Discriminator
from typing import Annotated, Dict, Literal, Optional, Sequence

from .features import PrecomputedFeatureConfig
from .precomputed_performance import performance_type

transferability_metric_names = Literal[
    "GBC",
    "LEEP",
    "Gaussian_LEEP",
    "Hscore",
    "Regularized_Hscore",
    "LogME",
    "NCTI",
    "CTE",
    "Transfer_Score",
    "Dispersion",
    "NuNo",
    "MaNo",
]


class TransferabilitySaveConfig(BaseModel):
    save_base_path: Optional[str]
    save_name: Optional[str]
    save_plot: bool


class TransferabilityMetricConfig(BaseModel):
    targets: Sequence[str]
    source_models: Dict[str, str]
    feature_config: PrecomputedFeatureConfig
    performance_config: Annotated[performance_type, Discriminator("name")]
    transferability_metrics: Sequence[transferability_metric_names]
    output_config: TransferabilitySaveConfig
