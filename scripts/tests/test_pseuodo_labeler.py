import torch
from typing import Dict, List, Any

from model_ranking import (
    BBBC039TargetConfig,
    InstanceSegmentationConfig,
    SemanticSegmentationConfig,
    Pytorch3DUnetModelConfig,
)
from model_ranking import (
    HammingDistanceEval,
    EffectiveInvarianceEval,
    CrossEntropyEval,
    DifferenceImageEval,
    EntropyEval,
    KLDivergenceEval,
    AdaptedRandErrorEval,
)
from model_ranking import (
    InputConsistencyPatchwisePseudoLabeler,
    ModelConsistencyPatchWisePseudoLabeler,
)
from model_ranking import add_device_to_config
from model_ranking import (
    get_model_path,
)
from pytorch3dunet.augment.transforms import (
    Transformer,
)
from pytorch3dunet.datasets.utils import (
    get_test_loaders,  # pyright: ignore[reportUnknownVariableType]
)
from pytorch3dunet.unet3d.model import (
    get_model,  # pyright: ignore[reportUnknownVariableType]
)
from pytorch3dunet.unet3d.utils import (
    load_checkpoint,  # pyright: ignore[reportUnknownVariableType]
)

transformer_config: Dict[str, List[Any]] = {
    "raw": [
        {
            "name": "AdditiveGaussianNoise",
            "execution_probability": 1,
            "scale": [0.1, 0.2],
        },
    ]
}

stats = {
    "pmin": None,
    "pmax": None,
    "mean": None,
    "std": None,
    "percentile_min": None,
    "percentile_max": None,
}

model_cfg: Dict[str, Any] = {
    "name": "UNet2D",
    "in_channels": 1,
    "out_channels": 1,
    "layer_order": "bcr",
    "f_maps": [32, 64, 128],
    "final_sigmoid": True,
    "feature_return": False,
    "is_segmentation": True,
    "feature_perturbation": None,
}

perturbed_model_config: Dict[str, Any] = {
    "name": "UNet2D",
    "in_channels": 1,
    "out_channels": 1,
    "layer_order": "bcr",
    "f_maps": [32, 64, 128],
    "final_sigmoid": True,
    "feature_return": False,
    "is_segmentation": True,
    "feature_perturbation": {
        "layers": [0],
        "random_seed": 42,
        "name": "DropOutPerturbation",
        "drop_rate": 0.2,
        "spatial_dropout": True,
    },
}
perturbed_model_cfg = Pytorch3DUnetModelConfig.model_validate(perturbed_model_config)

if __name__ == "__main__":

    target_cfg = BBBC039TargetConfig()
    pred_loader = target_cfg.loader

    pred_loader_cfg = pred_loader.create_config(
        output_dir="out_path",
        data_base_path="/scratch/talks/data",
    ).model_dump()
    config = {"loaders": pred_loader_cfg}
    config = add_device_to_config(config)

    transformer = Transformer(transformer_config, stats)
    model_path = get_model_path("BBBC039", "BC_model", "/g/kreshuk/talks/Models")
    model = get_model(model_cfg)
    load_checkpoint(model_path, model)
    model = model.cuda()
    model = model.eval()

    consis_metric = HammingDistanceEval(0.5)
    # consis_metric = EffectiveInvarianceEval()
    # consis_metric = CrossEntropyEval()
    # consis_metric = DifferenceImageEval()
    # consis_metric = EntropyEval()
    # consis_metric = KLDivergenceEval()
    # consis_metric = AdaptedRandErrorEval(dataset_name="BBBC039")

    instance_params = InstanceSegmentationConfig(
        name="instance",
        min_size=target_cfg.predictor_instance.min_size,
        zero_largest_instance=target_cfg.predictor_instance.zero_largest_instance,
        no_adjust_background=target_cfg.predictor_instance.no_adjust_background,
    )
    semantic_params = SemanticSegmentationConfig()

    pseudo_labeler = InputConsistencyPatchwisePseudoLabeler(
        transformer=transformer,
        consistency_metric=consis_metric,
        foreground_threshold=0.5,
        consistency_threshold=0.5,
        seg_params=semantic_params,
    )

    """
    pseudo_labeler = ModelConsistencyPatchWisePseudoLabeler(
        perturbed_model_config=perturbed_model_cfg,
        consistency_metric=consis_metric,
        foreground_threshold=0.5,
        consistency_threshold=0.5,
        seg_params=semantic_params,
    )
    """
    with torch.no_grad():
        for test_loader in get_test_loaders(config):
            img, _ = next(iter(test_loader))
            print(img.shape)
            img = torch.squeeze(img, dim=-3)
            img = img.cuda()
            pseudo_labels, label_mask = pseudo_labeler(model, img)
            print(f"label shape: {pseudo_labels.shape}")
            if label_mask is not None:
                print(f"label mask shape: {label_mask.shape}")
            break
