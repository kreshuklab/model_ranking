from model_ranking.dataclass import (
    SummaryResultsConfig,
)
from model_ranking.results import save_summary_metrics

results_config = SummaryResultsConfig(
    filter_patches=None,
    output_path="/g/kreshuk/talks/model_ranking/scripts/test_scripts/sweep2/BBBC039_to_BBBC039_gap/feature_perturbation_consistency/test/BC_IN_model2/norm_50_980/DO_a02/predictions",
    eval_key="AdaRandError",
    consis_key="ARE_th",
    overwrite_scores=False,
    save_select_patches=False,
)


save_summary_metrics(results_config)
