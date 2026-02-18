from typing_extensions import get_args
from pydantic import BaseModel, Discriminator
from typing import Annotated, Literal, Optional, Sequence, Tuple, Union

# from model_ranking.data_structures.constants import dataset_names
from model_ranking.data_structures.common_types import dataset_names

# from model_ranking.data_structures.common_types import feature_perturbation_type
from model_ranking.data_structures.perturbation import feature_perturbation_type

pytorch3dunet_model_names = Literal[
    "UNet2D", "UNet2d_as3d", "ResidualUNet2D", "ResidualUNet2D_as_3D", "UNet3D"
]

Unetr_model_names = Literal[
    "UnetrWrapper",
    "UnetrWithDropOut",
]

external_model_names = Literal[
    "Cellpose_SAM",
    "Micro_SAM",
    "SAM",
    "BioImageIO",
]

model_names = Union[pytorch3dunet_model_names, external_model_names, Unetr_model_names]


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


class SourceModelConfigBase(BaseModel):
    model_name: str
    model_type: model_names = "UNet2D"
    checkpoint_name: str = "best_checkpoint"


UNET2D_3LAYER_ARCHITECTURE = Pytorch3DUnetModelMetaConfig(
    name="UNet2D",
    in_channels=1,
    out_channels=1,
    layer_order="bcr",
    f_maps=(32, 64, 128),
    final_sigmoid=True,
    feature_return=False,
    is_segmentation=True,
)

UNET2D_4LAYER_ARCHITECTURE = Pytorch3DUnetModelMetaConfig(
    name="UNet2D",
    in_channels=1,
    out_channels=1,
    layer_order="bcr",
    f_maps=32,
    final_sigmoid=True,
    feature_return=False,
    is_segmentation=True,
)

RESIDUALUNET2D_5LAYER_ARCHITECTURE = Pytorch3DUnetModelMetaConfig(
    name="ResidualUNet2D",
    in_channels=1,
    out_channels=1,
    layer_order="bcr",
    f_maps=64,
    final_sigmoid=True,
    feature_return=False,
    is_segmentation=True,
)


UNETR_DEFAULT_ARCHITECTURE = UnetrModelMetaConfig(
    name="UnetrWrapper",
    in_channels=1,
    out_channels=1,
    img_size=256,  ### Place holder size will be overwritten on creation of UnetrModelConfig
    feature_size=16,
    hidden_size=768,
    mlp_dim=3072,
    num_heads=12,
    proj_type="conv",
    norm_name="batch",
    conv_block=True,
    res_block=True,
    dropout_rate=0.0,
    spatial_dims=2,
    qkv_bias=False,
    save_attn=False,
    is_segmentation=True,
    final_sigmoid=True,
)

UNETR_WITH_DROPOUT_ARCHITECTURE = UnetrWithDropOutModelMetaConfig(
    name="UnetrWithDropOut",
    in_channels=1,
    out_channels=1,
    img_size=256,  ### Place holder size will be overwritten on creation of UnetrModelConfig
    feature_size=16,
    hidden_size=768,
    mlp_dim=3072,
    num_heads=12,
    proj_type="conv",
    norm_name="batch",
    conv_block=True,
    res_block=True,
    spatial_dims=2,
    qkv_bias=False,
    save_attn=False,
    is_segmentation=True,
    final_sigmoid=True,
)


class ModelSourceConfig(SourceModelConfigBase):
    source_name: dataset_names

    def create_config(
        self,
        feature_perturbation: Optional[feature_perturbation_type],
    ):
        assert self.model_type in get_args(
            pytorch3dunet_model_names
        ), f"Invalid model type: {self.model_type}"
        if self.model_type == "UNet2D" or self.model_type == "UNet2d_as3d":
            if self.source_name in [
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
            ]:
                model = UNET2D_4LAYER_ARCHITECTURE.model_copy(
                    update={"name": self.model_type}
                )
            else:
                model = UNET2D_3LAYER_ARCHITECTURE.model_copy(
                    update={"name": self.model_type}
                )

        else:
            model = RESIDUALUNET2D_5LAYER_ARCHITECTURE.model_copy(
                update={"name": self.model_type}
            )
        return Pytorch3DUnetModelConfig(
            name=model.name,
            in_channels=model.in_channels,
            out_channels=model.out_channels,
            layer_order=model.layer_order,
            f_maps=model.f_maps,
            final_sigmoid=model.final_sigmoid,
            feature_return=model.feature_return,
            is_segmentation=model.is_segmentation,
            feature_perturbation=feature_perturbation,
        )

    def create_unetr_config(
        self,
        feature_perturbation: feature_perturbation_type,
        img_size: Union[Sequence[int], int],
    ):
        assert self.model_type in [
            "UnetrWrapper",
            "UnetrWithDropOut",
        ], f"Invalid model type for UnetrConfig: {self.model_type}"
        if self.model_type == "UnetrWrapper":
            model = UNETR_DEFAULT_ARCHITECTURE
            return UnetrModelConfig(
                name=model.name,
                in_channels=model.in_channels,
                out_channels=model.out_channels,
                img_size=img_size,
                feature_size=model.feature_size,
                hidden_size=model.hidden_size,
                mlp_dim=model.mlp_dim,
                num_heads=model.num_heads,
                proj_type=model.proj_type,
                norm_name=model.norm_name,
                conv_block=model.conv_block,
                res_block=model.res_block,
                dropout_rate=model.dropout_rate,
                spatial_dims=model.spatial_dims,
                qkv_bias=model.qkv_bias,
                save_attn=model.save_attn,
                is_segmentation=model.is_segmentation,
                final_sigmoid=model.final_sigmoid,
                feature_perturbation=feature_perturbation,
            )
        else:
            model = UNETR_WITH_DROPOUT_ARCHITECTURE
            return UnetrWithDropOutModelConfig(
                name=model.name,
                in_channels=model.in_channels,
                out_channels=model.out_channels,
                img_size=img_size,
                feature_size=model.feature_size,
                hidden_size=model.hidden_size,
                mlp_dim=model.mlp_dim,
                num_heads=model.num_heads,
                proj_type=model.proj_type,
                norm_name=model.norm_name,
                conv_block=model.conv_block,
                res_block=model.res_block,
                spatial_dims=model.spatial_dims,
                qkv_bias=model.qkv_bias,
                save_attn=model.save_attn,
                is_segmentation=model.is_segmentation,
                final_sigmoid=model.final_sigmoid,
                feature_perturbation=feature_perturbation,
            )


internal_model_type = Annotated[
    Union[Pytorch3DUnetModelConfig, UnetrModelConfig, UnetrWithDropOutModelConfig],
    Discriminator("name"),
]
