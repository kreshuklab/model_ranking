from collections import OrderedDict
import torch
import torch.nn as nn
import torchvision.models as models  # pyright: ignore[reportMissingTypeStubs]
from typing import Any, List
from torch.utils.hooks import RemovableHandle

from .dataclass import ClassificationModelConfig


class ResNet(nn.Module):
    def __init__(self, model_config: ClassificationModelConfig):
        super().__init__()
        self.ResNet = initialise_resNet(model_config)

    def forward(self, x: torch.Tensor):
        return self.ResNet(x)


def initialise_resNet(model_config: ClassificationModelConfig) -> nn.Module:
    if model_config.modelType == "ResNet18":
        model = models.resnet18()
    else:
        raise ValueError(f"Unknown model type: {model_config.modelType}")
    conv1_cfg = model_config.conv1
    model.conv1 = nn.Conv2d(
        conv1_cfg.in_channels,
        conv1_cfg.out_channels,
        kernel_size=conv1_cfg.kernel_size,
        stride=conv1_cfg.stride,
        padding=conv1_cfg.padding,
        bias=conv1_cfg.bias,
    )
    nr_filters = model.fc.in_features
    model.fc = nn.Linear(nr_filters, model_config.out_channels)
    return model


class ResNet18_classification(nn.Module):
    def __init__(self, model_config: ClassificationModelConfig):
        super().__init__()

        self.resnet_18 = initialise_resNet(model_config)
        self.forward_hooks: List[RemovableHandle] = []

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

    def remove_hooks(self):
        """Remove all registered forward hooks to prevent memory leaks."""
        for hook in self.forward_hooks:
            hook.remove()
        self.forward_hooks.clear()

    def __del__(self):
        """Cleanup hooks when the object is destroyed."""
        if hasattr(self, "forward_hooks"):
            self.remove_hooks()

    # forward pass
    def forward(self, x: torch.Tensor):
        out = self.resnet_18(x)
        return out

    def features(self, x: torch.Tensor):
        pred = self.forward(x)
        return pred, self.layer_acts
