from pathlib import Path
from pydantic import BaseModel, Discriminator
from typing import Annotated, Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

from model_ranking.dataclass import WandbConfig


class Conv1Config(BaseModel):
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int
    padding: int
    bias: bool


class ResNetConv1Config(Conv1Config):
    name: Literal["ResNet18", "ResNet50"]
    in_channels: int = 1
    out_channels: int = 64
    kernel_size: int = 7
    stride: int = 2
    padding: int = 3
    bias: bool = False


class DenseNetConv1Config(Conv1Config):
    name: Literal["DenseNet121", "DenseNet169"]
    in_channels: int = 1
    out_channels: int = 64
    kernel_size: int = 7
    stride: int = 2
    padding: int = 3
    bias: bool = False


class MobileNetV2Conv1Config(Conv1Config):
    name: Literal["MobileNetV2"]
    in_channels: int = 1
    out_channels: int = 32
    kernel_size: int = 3
    stride: int = 2
    padding: int = 1
    bias: bool = False


class MobileNetV3Conv1Config(Conv1Config):
    name: Literal["MobileNetV3"]
    in_channels: int = 1
    out_channels: int = 16
    kernel_size: int = 3
    stride: int = 2
    padding: int = 1
    bias: bool = False


class VGGConv1Config(Conv1Config):
    name: Literal["VGG16", "VGG19"]
    in_channels: int = 1
    out_channels: int = 64
    kernel_size: int = 3
    stride: int = 1
    padding: int = 1
    bias: bool = False


classification_conv1_type = Annotated[
    Union[
        ResNetConv1Config,
        DenseNetConv1Config,
        MobileNetV2Conv1Config,
        MobileNetV3Conv1Config,
        VGGConv1Config,
    ],
    Discriminator("name"),
]


class ClassificationModelConfig(BaseModel):
    conv1: classification_conv1_type
    out_channels: int
    modelname: str
    ckpt_path: Optional[str]
    ckpt_key: Optional[str]
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
    loader_rnd_seed: Optional[int] = None


class ClassificationOutputConfig(BaseModel):
    save_dir_path: Union[str, Path]
    prediction_threshold: float


class ClassificationPredictConfig(BaseModel):
    loader: ClassificationLoaderConfig
    model: ClassificationModelConfig
    output: ClassificationOutputConfig


class SchedulerConfig(BaseModel):
    mode: str = "min"
    factor: float = 0.5
    patience: int = 5


class LoggingSettings(BaseModel):
    log_image_interval: int
    log_val_images: bool
    log_pred: bool


class TrainingSettingsConfig(BaseModel):
    save_path: str
    num_epochs: int
    logging: LoggingSettings
    loss_function: Literal["BCEWithLogitsLoss"] = "BCEWithLogitsLoss"
    learning_rate: float = 1e-4
    scheduler_kwargs: SchedulerConfig = SchedulerConfig()


class ClassificationTrainConfig(BaseModel):
    wandb: WandbConfig
    train_loader: ClassificationLoaderConfig
    val_loader: ClassificationLoaderConfig
    test_loader: Optional[ClassificationLoaderConfig]
    model_cfg: ClassificationModelConfig
    training_config: TrainingSettingsConfig
