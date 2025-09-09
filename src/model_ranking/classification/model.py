import torch
import torch.nn as nn
import torchvision.models as models  # pyright: ignore[reportMissingTypeStubs]

from .dataclass import ClassificationModelConfig


class ClassificationNet(nn.Module):
    def __init__(self, model_config: ClassificationModelConfig):
        super().__init__()
        self.classification_net = initialise_classification_net(model_config)

    def forward(self, x: torch.Tensor):
        return self.classification_net(x)


def initialise_classification_net(model_config: ClassificationModelConfig) -> nn.Module:
    conv1_cfg = model_config.conv1
    input_conv = nn.Conv2d(
        conv1_cfg.in_channels,
        conv1_cfg.out_channels,
        kernel_size=conv1_cfg.kernel_size,
        stride=conv1_cfg.stride,
        padding=conv1_cfg.padding,
        bias=conv1_cfg.bias,
    )
    if conv1_cfg.name == "ResNet18":
        model = models.resnet18()
        model.conv1 = input_conv
        nr_filters = model.fc.in_features
        model.fc = nn.Linear(nr_filters, model_config.out_channels)
    elif conv1_cfg.name == "ResNet50":
        model = models.resnet50()
        model.conv1 = input_conv
        nr_filters = model.fc.in_features
        model.fc = nn.Linear(nr_filters, model_config.out_channels)
    elif conv1_cfg.name == "DenseNet121":
        model = models.densenet121()
        model.features.conv0 = input_conv
        nr_filters = model.classifier.in_features
        model.classifier = nn.Linear(nr_filters, model_config.out_channels)
    elif conv1_cfg.name == "DenseNet169":
        model = models.densenet169()
        model.features.conv0 = input_conv
        nr_filters = model.classifier.in_features
        model.classifier = nn.Linear(nr_filters, model_config.out_channels)
    elif conv1_cfg.name == "MobileNetV2":
        model = models.mobilenet_v2()
        model.features[0][0] = input_conv
        nr_filters = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(nr_filters, model_config.out_channels)
    elif conv1_cfg.name == "MobileNetV3":
        model = models.mobilenet_v3_small()
        model.features[0][0] = input_conv
        nr_filters = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(nr_filters, model_config.out_channels)
    elif conv1_cfg.name == "VGG16":
        model = models.vgg16()
        model.features[0] = input_conv  # pyright: ignore[reportIndexIssue]
        nr_filters = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(nr_filters, model_config.out_channels)
    elif conv1_cfg.name == "VGG19":
        model = models.vgg19()
        model.features[0] = input_conv  # pyright: ignore[reportIndexIssue]
        nr_filters = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(nr_filters, model_config.out_channels)
    else:
        raise NotImplementedError(f"Model {conv1_cfg.name} not implemented.")

    return model
