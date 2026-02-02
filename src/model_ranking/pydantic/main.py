from pydantic import BaseModel

from .data.eval_dataloaders import EvalDataloaderConfig
from .consistency_metrics import consistency_metric_type
from .performance_metrics import eval_metric_type


class EvaluateConfig(BaseModel, frozen=True):
    eval_dataloader: EvalDataloaderConfig
    eval_metric: eval_metric_type


class ConsistencyConfig(BaseModel, frozen=True):
    consistency_dataloader: EvalDataloaderConfig
    consistency_metric: consistency_metric_type
