from pydantic import BaseModel, Discriminator
from typing import Annotated, List, Literal, Optional, Union


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
    input_features: bool = False
    layer_key: Optional[str] = "ClassNet"
