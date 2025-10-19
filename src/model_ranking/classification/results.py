from tqdm import tqdm
import numpy as np
from typing import Dict

from .consistency import (
    ClassificationPredicitonLoadConfig,
)
from .utils import (
    get_classification_pred_path,
)

from model_ranking.utils import load_h5


def get_classification_performance_results(
    config: ClassificationPredicitonLoadConfig,
    performance_metric_key: str,
):
    per_target_performance: Dict[str, Dict[str, float]] = {}
    for tgt in config.target:
        per_model_performance: Dict[str, float] = {}
        for src_model in config.source:
            p_pred_path = get_classification_pred_path(
                model_name=src_model,
                target=tgt,
                base_path=config.base_path,
            )
            performance_score = load_h5(p_pred_path, performance_metric_key)
            per_model_performance[src_model] = float(performance_score)
        per_target_performance[tgt] = per_model_performance
    return per_target_performance


def get_classification_consistency_results(
    config: ClassificationPredicitonLoadConfig,
    consis_metric_key: str,
):
    per_perturbation_consis: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    for pert_type, pert_levels in config.perturbations.items():
        per_strength_consis: Dict[str, Dict[str, Dict[str, float]]] = {}
        for pert_level in tqdm(pert_levels):
            per_target_consis: Dict[str, Dict[str, float]] = {}
            for tgt in config.target:
                per_model_consis: Dict[str, float] = {}
                for src_model in config.source:
                    p_pred_path = get_classification_pred_path(
                        model_name=src_model,
                        target=tgt,
                        base_path=config.base_path,
                        aug=pert_type,
                        aug_str=pert_level,
                    )
                    consis_scores = load_h5(p_pred_path, consis_metric_key)
                    median_score = np.median(consis_scores, axis=1)[0]
                    per_model_consis[src_model] = median_score
                per_target_consis[tgt] = per_model_consis
            per_strength_consis[pert_level] = per_target_consis
        per_perturbation_consis[pert_type] = per_strength_consis
    return per_perturbation_consis
