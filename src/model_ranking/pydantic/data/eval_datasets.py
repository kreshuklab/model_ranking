from pydantic import BaseModel
from typing import Literal, Sequence, Optional, Union

from .datasets import (
    SBIAD1410PhaseConfig,
    TIFPhaseConfig,
    TIFtxtPhaseConfig,
    TIF_dataset_names,
)


class EvalDatasetConfig(BaseModel):
    name: Literal["StandardEvalDataset"]
    aug_name: str
    pred_path: Sequence[str]
    gt_path: Sequence[str]
    pred_key: str
    gt_key: str
    patch_key: str
    roi: Optional[Sequence[Sequence[int]]]
    ignore_index: Optional[int]
    ignore_path: Optional[str]
    ignore_key: Optional[str]
    convert_to_binary_label: bool
    convert_to_boundary_label: bool
    gt_zero_largest_instance: bool
    gt_zero_large_instances: bool
    gt_set_id_to_zero: Optional[int]
    min_object_size: Optional[int]
    zero_large_instances: bool
    zero_largest_instance: bool
    largest_obj_multiplier: Optional[float]
    max_obj_size: Optional[int]


class TIFEvalDatasetConfig(BaseModel):
    name: TIF_dataset_names
    eval: Union[TIFPhaseConfig, TIFtxtPhaseConfig]
    expand_dims: bool
    global_norm: bool
    percentiles: Optional[Sequence[Union[float, int]]]
    image_key: Optional[str]
    mask_key: Optional[str]
    min_object_size: Optional[int]
    zero_large_instances: bool


class SBIAD1410EvalDatasetConfig(BaseModel):
    name: Literal["S_BIAD1410_Dataset"]
    eval: SBIAD1410PhaseConfig
    global_normalization: bool
    global_percentiles: Optional[Sequence[Union[float, int]]]
    image_key: Optional[str]
    mask_key: Optional[str]
    zero_large_instances: bool = True
