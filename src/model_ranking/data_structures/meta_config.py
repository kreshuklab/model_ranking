from typing import Dict, List, Literal, Optional, Sequence, Tuple
from pydantic import BaseModel

from .ranking.consistency_metrics import consistency_metric_type
from .data.data import target_dataset_type
from .data.eval_dataloaders import EvalDataloaderConfig
from .ranking.performance_metrics import eval_metric_type
from .general.model import ModelSourceConfig, SourceModelConfigBase
from .general.summary import SummaryResultsMetaConfig
from .perturbation import FeaturePerturbationConfig
from .data.slice_builders import slice_builder_type

class EvaluateConfig(BaseModel, frozen=True):
    eval_dataloader: EvalDataloaderConfig
    eval_metric: eval_metric_type


class ConsistencyConfig(BaseModel, frozen=True):
    consistency_dataloader: EvalDataloaderConfig
    consistency_metric: consistency_metric_type


class OutputSettingsConfig(BaseModel):
    result_dir: Optional[str]
    approach: Optional[str]
    base_dir_path: str
    output_folder: Optional[str] = "norm"


run_mode_type = Literal[
    "full",
    "evaluation",
    "consistency",
    "pred_eval",
    "summary_results",
    "adaptive_batchnorm",
]


class MetaConfig(BaseModel):
    target_datasets: Sequence[target_dataset_type]
    source_models: Sequence[ModelSourceConfig]
    segmentation_mode: Literal["instance", "semantic"]
    run_mode: run_mode_type
    summary_results: SummaryResultsMetaConfig
    overwrite_yaml: bool
    data_base_path: str
    model_dir_path: str
    feature_perturbations: Optional[FeaturePerturbationConfig]
    output_settings: OutputSettingsConfig
    input_augs: Dict[str, List[Tuple[float, float]]]
    eval_settings: Optional[eval_metric_type]
    consistency_settings: Optional[consistency_metric_type]
    slice_builder_settings: Optional[slice_builder_type] = None
    data_fraction: Optional[float] = None
    foreground_ratio_threshold: Optional[float] = 0.0


class TransformerConsistencyMetaConfig(BaseModel):
    source_models: Sequence[SourceModelConfigBase]
    target_datasets: Sequence[target_dataset_type]
    data_base_path: str
    overwrite_yaml: bool
    input_augs: Dict[str, List[Tuple[float, float]]]
    summary_results: SummaryResultsMetaConfig
    output_settings: OutputSettingsConfig
    consistency_settings: Optional[consistency_metric_type]
