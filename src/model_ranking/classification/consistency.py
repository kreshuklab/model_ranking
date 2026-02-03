import numpy as np
from scipy.spatial.distance import hamming
from tqdm import tqdm

from model_ranking.utils import load_h5, save_h5

# from model_ranking.results import save_summary_metrics
# from model_ranking.dataclass import SummaryResultsConfig
from model_ranking.metrics import calculate_EI_binary

from .utils import (
    get_classification_pred_path,
)

from model_ranking.data_structures import (
    ClassificationConsistencyConfig,
)


def run_classification_consistency(config: ClassificationConsistencyConfig):
    consis_cfg = config.consistency_metric
    for src_model in config.source:
        for tgt in config.target:
            unp_pred_path = get_classification_pred_path(
                model_name=src_model,
                target=tgt,
                base_path=config.base_path,
                aug="None",
            )
            unp_preds = load_h5(unp_pred_path, "predictions")
            for pert_type, pert_levels in config.perturbations.items():
                for pert_level in tqdm(pert_levels):
                    p_pred_path = get_classification_pred_path(
                        model_name=src_model,
                        target=tgt,
                        base_path=config.base_path,
                        aug=pert_type,
                        aug_str=pert_level,
                    )
                    p_preds = load_h5(p_pred_path, "predictions")

                    # compute consistency metric
                    if consis_cfg.name == "EI":
                        preds = np.stack(
                            (
                                p_preds > consis_cfg.threshold,
                                unp_preds > consis_cfg.threshold,
                            )
                        )
                        preds_probs = np.stack((p_preds, unp_preds))
                        consis_score, _, _, _ = calculate_EI_binary(
                            preds,
                            preds_probs,
                        )
                    elif consis_cfg.name == "Hamming-Distance":
                        p_pred_th = (p_preds > consis_cfg.threshold).astype(np.uint8)
                        unp_pred_th = (unp_preds > consis_cfg.threshold).astype(
                            np.uint8
                        )
                        consis_score = np.array(hamming(p_pred_th, unp_pred_th))

                    else:
                        raise ValueError(
                            f"Unknown consistency metric: {consis_cfg.name}"
                        )

                    save_h5(
                        p_pred_path,
                        out_key=consis_cfg.save_key,
                        data=consis_score,
                        overwrite=consis_cfg.overwrite_scores,
                    )
                    # summary_cfg = SummaryResultsConfig(
                    #     filter_patches=None,
                    #     output_path=str(p_pred_path.parent),
                    #     eval_key=config.summary_results.eval_key,
                    #     consis_key=consis_cfg.save_key,
                    #     overwrite_scores=config.summary_results.overwrite_scores,
                    #     save_name_postfix=config.summary_results.save_name_postfix,
                    # )
                    # save_summary_metrics(summary_cfg)
