from monai.networks.nets import unetr
from ruyaml import Any
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Sequence, Union

from pytorch3dunet.unet3d.utils import (
    get_class,  # pyright:ignore[reportUnknownVariableType]
)


class UnetrWrapper(unetr.UNETR):
    """
    Wrapper for the monai UNETR model to be compatible with pytorch3dunet.
    This class is used to ensure that the model can be used with the pytorch3dunet
    training and prediction code.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: Sequence[int] | int,
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        proj_type: str = "conv",
        norm_name: Union[Tuple[str, str], str] = "instance",
        conv_block: bool = True,
        res_block: bool = True,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        qkv_bias: bool = False,
        save_attn: bool = False,
        is_segmentation: bool = True,
        final_sigmoid: bool = True,
        feature_perturbation: Optional[Dict[str, Any]] = None,
        **kwargs,  # pyright: ignore
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=img_size,
            feature_size=feature_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
            proj_type=proj_type,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
            qkv_bias=qkv_bias,
            save_attn=save_attn,
        )

        if is_segmentation:
            # semantic segmentation problem
            if final_sigmoid:
                self.final_activation = nn.Sigmoid()
            else:
                self.final_activation = nn.Softmax(dim=1)
        else:
            # regression problem
            self.final_activation = None

        if feature_perturbation is not None:
            perturbation_class = get_class(
                feature_perturbation["name"],
                modules=["pytorch3dunet.unet3d.feature_perturbation"],
            )
            self.feature_perturbation = perturbation_class(**feature_perturbation)
        else:
            self.feature_perturbation = feature_perturbation

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        x_in = x_in.squeeze(2)

        x, hidden_states_out = self.vit(x_in)
        enc1 = self.encoder1(x_in)
        x2 = hidden_states_out[3]
        enc2 = self.encoder2(self.proj_feat(x2))
        x3 = hidden_states_out[6]
        enc3 = self.encoder3(self.proj_feat(x3))
        x4 = hidden_states_out[9]
        enc4 = self.encoder4(self.proj_feat(x4))

        if self.feature_perturbation is not None:
            x = self.feature_perturbation(x)

        dec4 = self.proj_feat(x)  # pyright: ignore[reportUnknownVariableType]
        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        out = self.decoder2(dec1, enc1)
        out = self.out(out)

        # out = super().forward(x_in).unsqueeze(2)
        # apply final_activation (i.e. Sigmoid or Softmax) only during prediction.
        # During training the network outputs logits
        if not self.training and self.final_activation is not None:
            out = self.final_activation(out)
        return out.unsqueeze(2)
