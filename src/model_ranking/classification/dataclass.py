from pathlib import Path
from pydantic import BaseModel
from typing import Any, List, Literal, Optional, Sequence, Tuple, Union


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


class TTAugmentationsConfig(BaseModel):
    aug_rnd_seed: int
    transform_params: Optional[Sequence[Sequence[Any]]]
    raw_transform_params: Optional[Sequence[Sequence[Any]]]
    ndim: int
    n_samples: int


class ClassificationFilteredDatasetConfig(BaseModel):
    raw_path: Union[str, Path]
    raw_key: str
    mask_path: Union[str, Path]
    mask_key: str
    patch_shape: Tuple[int, int, int]
    repeat_patches: bool
    patch_rnd_seed: int
    mask_return_mode: bool
    patch_return_mode: bool


class ClassificationTTALoaderConfig(BaseModel):
    patch_position: ClassificationPatchPositionConfig
    TTAugmentations: Optional[TTAugmentationsConfig]
    dataset: ClassificationFilteredDatasetConfig
    n_samples: int
    ndim: int
    batch_size: int
    shuffle: bool
    num_workers: int
