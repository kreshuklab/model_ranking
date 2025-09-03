from pathlib import Path
from pydantic import BaseModel
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union


class Conv1Config(BaseModel):
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int
    padding: int
    bias: bool


class ClassificationModelConfig(BaseModel):
    conv1: Conv1Config
    out_channels: int
    modelname: str
    modelType: Literal["ResNet18"]
    ckpt_path: Union[str, Path]
    ckpt_key: str
    feature_layers: Optional[Union[List[str], List[int]]]


class ClassificationPatchPositionConfig(BaseModel):
    patch_pos_path: str
    patch_pos_key: str
    roi: Optional[Sequence[Sequence[int]]]
    rnd_seed: Optional[int]
    n_unique_patches: Optional[int]
    slice_offset: Optional[int]


class AugmentationsConfig(BaseModel):
    transform_params: Optional[Sequence[Dict[str, Any]]]
    raw_transform_params: Optional[Sequence[Dict[str, Any]]]
    ndim: int


class ClassificationFilteredDatasetConfig(BaseModel):
    raw_path: Union[str, Path]
    raw_key: str
    mask_path: Union[str, Path]
    mask_key: str
    patch_shape: Tuple[int, int, int]
    repeat_patches: bool
    sample_patches: bool
    patch_rnd_seed: Optional[int]
    mask_return_mode: bool
    patch_return_mode: bool


class ClassificationLoaderConfig(BaseModel):
    patch_position: ClassificationPatchPositionConfig
    aug_config: Optional[AugmentationsConfig]
    dataset: ClassificationFilteredDatasetConfig
    n_samples: Optional[int]
    ndim: int
    batch_size: int
    shuffle: bool
    num_workers: int


class ClassificationOutputConfig(BaseModel):
    save_dir_path: Union[str, Path]
    prediction_threshold: float


class ClassificationPredictConfig(BaseModel):
    loader: ClassificationLoaderConfig
    model: ClassificationModelConfig
    output: ClassificationOutputConfig
