from model_ranking.consistency import (
    # get_consistency_loaders,
    # calc_consistency_score,
    run_consistency_evaluation,
)
from model_ranking.dataclass import (
    AdaptedRandErrorConfig,
    ConsistencyConfig,
    ConsistencyMetaConfig,
    # Eval_TIF_TxtDataloaderMetaConfig,
    BBBC039TargetConfig,
)

dataloader_config = BBBC039TargetConfig.model_fields[
    "consis_dataloader_instance"
].default.create_consis_config(
    perturbed_dir=(
        "/g/kreshuk/talks/model_ranking/scripts/test_scripts/BBBC039_to_BBBC039_gap/feature_perturbation_consistency/test/BC_model4/norm_50_980/DO_a01/predictions",
    ),
    unperturbed_dir=(
        "/g/kreshuk/talks/model_ranking/scripts/test_scripts/BBBC039_to_BBBC039_gap/feature_perturbation_consistency/test/BC_model4/norm_50_980/none/predictions",
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

scores, masks = run_consistency_evaluation(consis_config)

"""
for loader in get_consistency_loaders(dataloader_config):
    scores, mask = calc_consistency_score(
        loader,
        consis_config,
    )
"""
