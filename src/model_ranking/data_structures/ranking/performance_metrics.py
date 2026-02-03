import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Discriminator
import torch
from typing import Annotated, Any, List, Literal, Optional, Union

from model_ranking.metrics import (
    MultiClassF1Eval,
    BinaryF1Eval,
    SoftF1Eval,
    BinaryAccuracyEval,
    AdaptedRandErrorEval,
)

from pytorch3dunet.unet3d.metrics import (
    InstanceAveragePrecision,
)


class EvalMetricConfig(BaseModel, frozen=True):
    eval_save_key: str
    overwrite_score: bool


class AdaptedRandErrorConfig(BaseModel, frozen=True):
    name: Literal["AdaptedRandError"] = "AdaptedRandError"
    num_dilations: Optional[int] = 1
    num_erosions: Optional[int] = 1

    def initialise_metric(self, incomplete_gt: bool) -> AdaptedRandErrorEval:
        return AdaptedRandErrorEval(
            incomplete_gt=incomplete_gt,
            num_dilations=self.num_dilations,
            num_erosions=self.num_erosions,
        )

    def initialise_score(self, num_samples: int) -> torch.Tensor:
        return torch.zeros((num_samples, 3), dtype=torch.float32)

    def initialise_score_array(self, num_samples: int) -> NDArray[Any]:
        return np.zeros((num_samples, 3), dtype=np.float32)


class AdaptedRandErrorEvalConfig(AdaptedRandErrorConfig, EvalMetricConfig, frozen=True):
    pass


class MeanAvgPrecisionConfig(BaseModel, frozen=True):
    name: Literal["MeanAvgPrecision"] = "MeanAvgPrecision"
    iou_range: Optional[List[Union[float, int]]] = [0.5, 0.95, 10]
    min_instance_size: Optional[int] = None

    def initialise_metric(self) -> InstanceAveragePrecision:
        return InstanceAveragePrecision(
            min_instance_size=self.min_instance_size,
            iou_range=self.iou_range,
        )

    def initialise_score(self, num_samples: int) -> torch.Tensor:
        return torch.zeros(num_samples, dtype=torch.float32)

    def initialise_score_array(self, num_samples: int) -> NDArray[Any]:
        return np.zeros(num_samples, dtype=np.float32)


class MeanAvgPrecisionEvalConfig(MeanAvgPrecisionConfig, EvalMetricConfig, frozen=True):
    pass


class MultiClassF1Config(EvalMetricConfig, frozen=True):
    name: Literal["MultiClassF1"] = "MultiClassF1"
    threshold: float = 0.5

    def initialise_metric(self) -> MultiClassF1Eval:
        return MultiClassF1Eval(
            threshold=self.threshold,
        )

    def initialise_score(self, num_samples: int) -> torch.Tensor:
        return torch.zeros((num_samples, 2), dtype=torch.float32)


class BinaryAccuracyConfig(EvalMetricConfig, frozen=True):
    name: Literal["BinaryAccuracy"] = "BinaryAccuracy"
    threshold: float = 0.5

    def initialise_metric(self) -> BinaryAccuracyEval:
        return BinaryAccuracyEval(
            threshold=self.threshold,
        )

    def initialise_score(self, num_samples: int) -> torch.Tensor:
        return torch.zeros(num_samples, dtype=torch.float32)


class BinaryF1Config(EvalMetricConfig, frozen=True):
    name: Literal["BinaryF1"] = "BinaryF1"
    threshold: float = 0.5

    def initialise_metric(self) -> BinaryF1Eval:
        return BinaryF1Eval(
            threshold=self.threshold,
        )

    def initialise_score(self, num_samples: int) -> torch.Tensor:
        return torch.zeros(num_samples, dtype=torch.float32)


class SoftF1Config(EvalMetricConfig, frozen=True):
    name: Literal["SoftF1"] = "SoftF1"

    def initialise_metric(self) -> SoftF1Eval:
        return SoftF1Eval()

    def initialise_score(self, num_samples: int) -> torch.Tensor:
        return torch.zeros(num_samples, dtype=torch.float32)


eval_metric_type = Annotated[
    Union[
        AdaptedRandErrorEvalConfig,
        BinaryF1Config,
        BinaryAccuracyConfig,
        MeanAvgPrecisionEvalConfig,
        MultiClassF1Config,
        SoftF1Config,
    ],
    Discriminator("name"),
]
