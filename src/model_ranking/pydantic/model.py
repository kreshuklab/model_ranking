from pydantic import BaseModel, Discriminator
from typing import Annotated, Literal, Optional, Sequence, Tuple, Union

from .perturbation import feature_perturbation_type

pytorch3dunet_model_names = Literal[
    "UNet2D", "UNet2d_as3d", "ResidualUNet2D", "ResidualUNet2D_as_3D", "UNet3D"
]


class Pytorch3DUnetModelMetaConfig(BaseModel, frozen=True):
    name: pytorch3dunet_model_names
    in_channels: int
    out_channels: int
    layer_order: str
    f_maps: Union[int, Sequence[int]]
    final_sigmoid: bool
    feature_return: bool
    is_segmentation: Optional[bool]


class UnetrModelMetaConfig(BaseModel, frozen=True):
    name: Literal["UnetrWrapper"]
    in_channels: int
    out_channels: int
    img_size: Union[Sequence[int], int]
    feature_size: int
    hidden_size: int
    mlp_dim: int
    num_heads: int
    proj_type: str
    norm_name: Union[Tuple[str, ...], str]
    conv_block: bool
    res_block: bool
    dropout_rate: float
    spatial_dims: int
    qkv_bias: bool
    save_attn: bool
    is_segmentation: bool
    final_sigmoid: bool


class UnetrWithDropOutModelMetaConfig(BaseModel, frozen=True):
    name: Literal["UnetrWithDropOut"]
    in_channels: int
    out_channels: int
    img_size: Union[Sequence[int], int]
    feature_size: int
    hidden_size: int
    mlp_dim: int
    num_heads: int
    proj_type: str
    norm_name: Union[Tuple[str, ...], str]
    conv_block: bool
    res_block: bool
    spatial_dims: int
    qkv_bias: bool
    save_attn: bool
    is_segmentation: bool
    final_sigmoid: bool


class UnetrModelConfig(UnetrModelMetaConfig, frozen=True):
    feature_perturbation: Annotated[feature_perturbation_type, Discriminator("name")]


class UnetrWithDropOutModelConfig(UnetrWithDropOutModelMetaConfig, frozen=True):
    feature_perturbation: Annotated[feature_perturbation_type, Discriminator("name")]


class Pytorch3DUnetModelConfig(Pytorch3DUnetModelMetaConfig, frozen=True):
    feature_perturbation: Annotated[feature_perturbation_type, Discriminator("name")]
