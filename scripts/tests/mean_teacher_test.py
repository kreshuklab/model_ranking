from typing import Dict, List, Any

from model_ranking import (
    InputConsisPseudoLabelerConfig,
    SemanticSegmentationConfig,
    Pytorch3DUnetModelConfig,
    HammingDistanceConfig,
    # consistency_metric_type,
    # ModelConsisPseudoLabelerConfig,
)
from model_ranking import run_mean_teacher

output_path = "/g/kreshuk/talks/model_ranking/tests/mean_teacher2"
unsupervised_train_paths = [
    "/g/kreshuk/talks/data/epfl/resized_pixels/train.h5",
]
unsupervised_val_paths = [
    "/g/kreshuk/talks/data/epfl/resized_pixels/val.h5",
]

source_checkpoint = "/g/kreshuk/talks/segmentation_ModelSelection/experiments/Hmito/BatchNorm/Hm_model3/best_checkpoint.pytorch"
patch_shape = (1, 64, 64)

transformer_cfg: Dict[str, List[Any]] = {
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

model_config: Dict[str, Any] = {
    "name": "UNet2d_as3d",
    "in_channels": 1,
    "out_channels": 1,
    "layer_order": "bcr",
    "f_maps": 32,
    "final_sigmoid": True,
    "feature_return": False,
    "is_segmentation": True,
    "feature_perturbation": None,
}

model_cfg = Pytorch3DUnetModelConfig.model_validate(model_config)

consis_config: Dict[str, Any] = {
    "name": "Hamming-Distance",
    "save_key": None,
    "save_mask": None,
    "mask_threshold": 0.5,
    "threshold": 0.5,
    "overwrite_score": None,
}

consis_cfg = HammingDistanceConfig.model_validate(consis_config)

pseudo_labeler_cfg = InputConsisPseudoLabelerConfig(
    name="input_consistency",
    consistency_metric=consis_cfg,
    consistency_threshold=0.5,
    seg_params=SemanticSegmentationConfig(),
    transformer_cfg=transformer_cfg,
    stats_cfg=stats,
)


run_mean_teacher(
    name="MT_test",
    output_root_path=output_path,
    unsupervised_train_paths=unsupervised_train_paths,
    unsupervised_val_paths=unsupervised_val_paths,
    patch_shape=patch_shape,
    pseudo_labeler_config=pseudo_labeler_cfg,
    model_config=model_cfg,
    source_checkpoint=source_checkpoint,
    supervised_train_paths=None,
    supervised_val_paths=None,
    raw_key="raw",
    raw_key_supervised=None,
    label_key=None,
    batch_size=1,
    lr=1e-4,
    n_iterations=3,
    n_samples_train=5,
    n_samples_val=5,
)
