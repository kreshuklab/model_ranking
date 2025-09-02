from collections import OrderedDict
import torch
import torch.nn as nn
import torchvision.models as models  # pyright: ignore[reportMissingTypeStubs]
from typing import Any

from .dataclass import ClassificationModelConfig


def initialise_resNet18(model_config: ClassificationModelConfig) -> nn.Module:
    resnet_18 = models.resnet18()
    conv1_cfg = model_config.conv1
    resnet_18.conv1 = nn.Conv2d(
        conv1_cfg.in_channels,
        conv1_cfg.out_channels,
        kernel_size=conv1_cfg.kernel_size,
        stride=conv1_cfg.stride,
        padding=conv1_cfg.padding,
        bias=conv1_cfg.bias,
    )
    nr_filters = resnet_18.fc.in_features
    resnet_18.fc = nn.Linear(nr_filters, model_config.out_channels)
    return resnet_18


class ResNet18_classification(nn.Module):
    def __init__(self, model_config: ClassificationModelConfig):
        super().__init__()

        self.resnet_18 = initialise_resNet18(model_config)
        self.forward_hooks = []

        # register hooks
        if model_config.feature_layers:
            self.layer_acts: OrderedDict[str, Any] = OrderedDict()
            for layer_num, layer_name in enumerate(self.resnet_18._modules.keys()):
                if layer_num in model_config.feature_layers:
                    self.forward_hooks.append(
                        getattr(self.resnet_18, layer_name).register_forward_hook(
                            self.get_activations(layer_name)
                        )
                    )

    # Defining hook to get intermediate features
    def get_activations(self, layer_name: str):
        def hook(module: nn.Module, input: Any, output: Any):
            self.layer_acts[layer_name] = output

        return hook

    # forward pass
    def forward(self, x: torch.Tensor):
        out = self.resnet_18(x)
        return out

    def features(self, x: torch.Tensor):
        pred = self.forward(x)
        return pred, self.layer_acts
