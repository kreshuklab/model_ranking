"""
Shared type definitions used across data_structures modules.

This module contains type definitions that are used by multiple submodules
to avoid circular dependencies.
"""

from pydantic import BaseModel
from typing import Literal, Optional, Sequence


class ForegroundFilterConfig(BaseModel):
    """Configuration for foreground filtering in data processing."""

    name: Literal["ForegroundFilter"]
    foreground_threshold: float
    gt_dir_path: str
    gt_key: Optional[str]
    roi: Optional[Sequence[Sequence[int]]]
    save_selection: bool
    overwrite: bool

    def create_config(self, data_base_path: str):
        gt_dir_path = data_base_path + self.gt_dir_path
        return ForegroundFilterConfig(
            name=self.name,
            foreground_threshold=self.foreground_threshold,
            gt_dir_path=gt_dir_path,
            gt_key=self.gt_key,
            roi=self.roi,
            save_selection=self.save_selection,
            overwrite=self.overwrite,
        )


dataset_names = Literal[
    "Go-Nuclear",
    "S_BIAD1196",
    "S_BIAD1410",
    "FlyWing",
    "Ovules",
    "PNAS",
    "EPFL",
    "Hmito",
    "Rmito",
    "VNC",
    "BBBC039",
    "DSB2018",
    "HeLaNuc",
    "Hoechst",
    "S_BIAD634",
    "S_BIAD895",
    "Covid_IF",
]


# class FeaturePerturbationBaseConfig(BaseModel):
#     layers: Sequence[int]
#     random_seed: int


# class DropOutPerturbationConfig(FeaturePerturbationBaseConfig):
#     name: Literal["DropOutPerturbation"]
#     drop_rate: float
#     spatial_dropout: bool


# class FeatureDropPerturbationConfig(FeaturePerturbationBaseConfig):
#     name: Literal["FeatureDropPerturbation"]
#     lower_th: float
#     upper_th: float


# class FeatureNoisePerturbationConfig(FeaturePerturbationBaseConfig):
#     name: Literal["FeatureNoisePerturbation"]
#     uniform_range: float


# feature_perturbation_type = Optional[
#     Union[
#         DropOutPerturbationConfig,
#         FeatureDropPerturbationConfig,
#         FeatureNoisePerturbationConfig,
#     ]
# ]
