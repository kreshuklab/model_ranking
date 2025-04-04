from model_ranking.consistency import (
    get_consistency_loaders,
    calc_consistency_score,
)
from model_ranking.dataclass import (
    AdaptedRandErrorConfig,
    ConsistencyConfig,
    ConsistencyMetaConfig,
    Eval_TIF_TxtDataloaderMetaConfig,
)

consis_dataloader = Eval_TIF_TxtDataloaderMetaConfig(
    batch_size=1,
    num_workers=8,
    name="TIF_txt_Dataset",
    expand_dims=True,
    global_norm=False,
    percentiles=None,
    image_key="segmentation",
    min_object_size=50,
    instance_zero_background=False,
    mask_dir=(
        "/g/kreshuk/talks/model_ranking/scripts/test_scripts/BBBC039_to_BBBC039_gap/feature_perturbation_consistency/test/BC_model4/norm_50_980/none/predictions",
    ),
    mask_key="segmentation",
    filenames_path="/BBBC039/test.txt",
    transformer={
        "raw": [],
        "label": [
            {"name": "CropToFixed", "size": (512, 512), "centered": True},
        ],
    },
)

dataloader_config = consis_dataloader.create_config(
    image_dir=(
        "/g/kreshuk/talks/model_ranking/scripts/test_scripts/BBBC039_to_BBBC039_gap/feature_perturbation_consistency/test/BC_model4/norm_50_980/DO_a01/predictions",
    ),
    data_base_path="/scratch/talks/data",
)

consis_config = ConsistencyConfig(
    consistency_dataloader=dataloader_config,
    consistency_metric=AdaptedRandErrorConfig(),
    consistency_settings=ConsistencyMetaConfig(
        save_key="ARE_score",
        save_mask=True,
        mask_threshold=0.5,
        overwrite_score=False,
    ),
)

for loader in get_consistency_loaders(dataloader_config):
    scores, mask = calc_consistency_score(
        loader,
        consis_config,
    )
