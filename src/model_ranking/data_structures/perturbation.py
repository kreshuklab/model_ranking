from pydantic import BaseModel
from typing import Dict, Literal, Optional, Sequence, Tuple, Union

# from model_ranking.data_structures.common_types import (
#     FeaturePerturbationBaseConfig,
#     DropOutPerturbationConfig,
#     FeatureDropPerturbationConfig,
#     FeatureNoisePerturbationConfig,
#     feature_perturbation_type,
# )


class FeaturePerturbationBaseConfig(BaseModel):
    layers: Sequence[int]
    random_seed: int


class DropOutPerturbationConfig(FeaturePerturbationBaseConfig):
    name: Literal["DropOutPerturbation"]
    drop_rate: float
    spatial_dropout: bool


class FeatureDropPerturbationConfig(FeaturePerturbationBaseConfig):
    name: Literal["FeatureDropPerturbation"]
    lower_th: float
    upper_th: float


class FeatureNoisePerturbationConfig(FeaturePerturbationBaseConfig):
    name: Literal["FeatureNoisePerturbation"]
    uniform_range: float


feature_perturbation_type = Optional[
    Union[
        DropOutPerturbationConfig,
        FeatureDropPerturbationConfig,
        FeatureNoisePerturbationConfig,
    ]
]


class FeaturePerturbationConfig(BaseModel):
    perturbation_types: Sequence[
        Literal[
            "DropOutPerturbation",
            "FeatureDropPerturbation",
            "FeatureNoisePerturbation",
            "None",
        ]
    ]
    layers: Sequence[int]
    dropOut_rates: Optional[Sequence[float]]
    spatial_dropout: Optional[bool]
    featureDrop_thresholds: Optional[Sequence[Tuple[float, float]]]
    featureNoise_ranges: Optional[Sequence[float]]
    random_seed: int


class InputPerturbationConfig(BaseModel):
    # name: Literal["RandomGamma", "RandomBrightness", "RandomContrast"]
    name: str
    execution_probability: float
    alpha: Tuple[float, float]
    clip_kwargs: Optional[Dict[str, float]]


class InputGaussianConfig(BaseModel):
    # name: Literal["AdditiveGaussianNoise"]
    name: str
    execution_probability: float
    scale: Tuple[float, float]
