import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Discriminator
from typing import Annotated, Any, Literal, List, Optional, Dict, Sequence, Union

from model_ranking.metrics import (
    DifferenceImageEval,
    EffectiveInvarianceEval,
    EntropyEval,
    KLDivergenceEval,
    CrossEntropyEval,
    HammingDistanceEval,
)

from .performance_metrics import AdaptedRandErrorConfig, MeanAvgPrecisionConfig

consistency_metric_names = Literal[
    "Diff",
    "EI",
    "Entropy",
    "Cross-Entropy",
    "KL-Divergence",
    "Hamming-Distance",
    "Rand-Index",
    "Adapted-Rand-Error",
    "AdaRand-Error",
    "MeanAvgPrecision",
]


class ConsistencyMetricConfig(BaseModel):
    metric: consistency_metric_names
    threshold: List[float]
    pred_key: str
    save_key: str
    entr_base: Optional[int]
    diff_alpha: Optional[float]
    select_alphas: Optional[Dict[str, List[str]]]
    save_mask: bool
    ignore_path: Optional[str]
    ignore_key: Optional[str]
    border_parameters: Optional[Dict[str, int]]
    remove_background: bool
    zero_largest_instance: bool


class ConsistencyMetaConfig(BaseModel, frozen=True):
    save_key: Optional[str]
    save_mask: Optional[bool]
    mask_threshold: float
    overwrite_score: bool
    bckg_consistency: bool = False


class AdaptedRandErrorConsisConfig(
    AdaptedRandErrorConfig, ConsistencyMetaConfig, frozen=True
):
    pass


class MeanAvgPrecisionConsisConfig(
    MeanAvgPrecisionConfig, ConsistencyMetaConfig, frozen=True
):
    pass


class DifferenceImageConfig(ConsistencyMetaConfig, frozen=True):
    name: Literal["Diff"] = "Diff"
    diff_alpha: float = 1

    def initialise_metric(self) -> DifferenceImageEval:
        return DifferenceImageEval(
            diff_alpha=self.diff_alpha,
        )

    def initialise_score(
        self, num_samples: int, sample_shape: Sequence[int]
    ) -> NDArray[Any]:
        return np.zeros((num_samples, *sample_shape), dtype=np.float32)


class EffectiveInvarianceConfig(ConsistencyMetaConfig, frozen=True):
    name: Literal["EI"] = "EI"
    threshold: float = 0.5

    def initialise_metric(self) -> EffectiveInvarianceEval:
        return EffectiveInvarianceEval(
            threshold=self.threshold,
            invert_threshold=self.bckg_consistency,
        )

    def initialise_score(
        self, num_samples: int, sample_shape: Sequence[int]
    ) -> NDArray[Any]:
        return np.zeros((num_samples, *sample_shape), dtype=np.float32)


class EntropyConfig(ConsistencyMetaConfig, frozen=True):
    name: Literal["Entropy"] = "Entropy"
    entr_base: int = 2

    def initialise_metric(self) -> EntropyEval:
        return EntropyEval(
            entr_base=self.entr_base,
        )

    def initialise_score(
        self, num_samples: int, sample_shape: Sequence[int]
    ) -> NDArray[Any]:
        return np.zeros((num_samples, *sample_shape), dtype=np.float32)


class KLDivergenceConfig(ConsistencyMetaConfig, frozen=True):
    name: Literal["KL-Divergence"] = "KL-Divergence"
    eps: float = 1e-7
    entr_base: int = 2

    def initialise_metric(self) -> KLDivergenceEval:
        return KLDivergenceEval(
            eps=self.eps,
            entr_base=self.entr_base,
        )

    def initialise_score(
        self, num_samples: int, sample_shape: Sequence[int]
    ) -> NDArray[Any]:
        return np.zeros((num_samples, *sample_shape), dtype=np.float32)


class CrossEntropyConfig(ConsistencyMetaConfig, frozen=True):
    name: Literal["Cross-Entropy"] = "Cross-Entropy"
    eps: float = 1e-7
    entr_base: int = 2

    def initialise_metric(self) -> CrossEntropyEval:
        return CrossEntropyEval(
            eps=self.eps,
            entr_base=self.entr_base,
        )

    def initialise_score(
        self, num_samples: int, sample_shape: Sequence[int]
    ) -> NDArray[Any]:
        return np.zeros((num_samples, *sample_shape), dtype=np.float32)


class HammingDistanceConfig(ConsistencyMetaConfig, frozen=True):
    name: Literal["Hamming-Distance"] = "Hamming-Distance"
    threshold: float = 0.5

    def initialise_metric(self) -> HammingDistanceEval:
        return HammingDistanceEval(
            threshold=self.threshold,
            invert_threshold=self.bckg_consistency,
        )

    def initialise_score(self, num_samples: int) -> NDArray[Any]:
        return np.zeros(num_samples, dtype=np.float32)


consistency_metric_type = Annotated[
    Union[
        AdaptedRandErrorConsisConfig,
        MeanAvgPrecisionConsisConfig,
        CrossEntropyConfig,
        DifferenceImageConfig,
        EffectiveInvarianceConfig,
        EntropyConfig,
        HammingDistanceConfig,
        KLDivergenceConfig,
    ],
    Discriminator("name"),
]
