import json
import numpy as np
import os
from pathlib import Path
from pydantic import BaseModel
import typer
from typing import Annotated, Any, Dict, List

from pytorch3dunet.unet3d.config import (
    load_config_direct,  # pyright: ignore[reportUnknownVariableType]
)

from model_ranking import get_output_dir_paths
from model_ranking.transferability_metrics.SEG import (
    get_weights,
    get_f1_scores,  # pyright: ignore
)


class SEGConfig(BaseModel):
    methods: List[str]
    target: str
    result_dir: str
    base_path: str
    save_dir: str
    agree_ratios: List[float]
    radii: List[int]
    output_name: str


# methods = [
#     "BC_IN_model2",
#     "HN_IN_model2",
#     "Hst_IN_model3",
#     "895_IN_model2",
#     "1410_IN_model1",
# ]
# target = "Hoechst"
# result_dir = "P1"
# base_path = "/g/kreshuk/talks/consistency_results/Instance_segmentation/nuclei"
# save_dir = ""

# agree_ratios = [0.3]
# radii = [30]


def main(
    config: Annotated[str, typer.Option(help="Path to the configuration file")],
):
    cfg_data, _ = load_config_direct(config)

    cfg = SEGConfig.model_validate(cfg_data)

    model_pred_paths: Dict[str, Path] = {}
    for model in cfg.methods:
        paths = get_output_dir_paths(
            cfg.target,
            model,
            cfg.result_dir,
            base_path=cfg.base_path,
            specfic_perturbation="none",
        )
        assert (
            len(paths) == 1
        ), f"Expected exactly one path for model {model}, found {len(paths)}"
        model_pred_paths[model] = paths[0]

    sample_ids = sorted(
        [
            Path(f).stem
            for f in os.listdir(model_pred_paths[cfg.methods[0]])
            if f.endswith(".h5")
        ]
    )

    equal_weights = np.ones((len(sample_ids), len(cfg.methods))) * (
        1 / len(cfg.methods)
    )

    weights = {}
    eq_wgt_f1s = {}
    uneq_wgt_f1s = {}

    for r in cfg.radii:
        per_AR_weights = {}
        per_AR_eq_f1s = {}
        per_AR_uneq_f1s = {}

        for agree_ratio in cfg.agree_ratios:
            unequal_weights = get_weights(
                model_pred_paths, sample_ids, cfg.methods, r, agree_ratio=agree_ratio
            )
            per_AR_weights[agree_ratio] = unequal_weights
            per_AR_eq_f1s[agree_ratio] = get_f1_scores(
                model_pred_paths, sample_ids, cfg.methods, equal_weights, agree_ratio, r
            )
            per_AR_uneq_f1s[agree_ratio] = get_f1_scores(
                model_pred_paths,
                sample_ids,
                cfg.methods,
                unequal_weights,
                agree_ratio,
                r,
            )

        weights[r] = per_AR_weights
        eq_wgt_f1s[r] = per_AR_eq_f1s
        uneq_wgt_f1s[r] = per_AR_uneq_f1s

    save_path = Path(cfg.save_dir) / f"to_{cfg.target}"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(os.path.join(save_path, f"{cfg.output_name}.json"), "w") as f:
        results: Dict[str, Any] = {
            "methods": cfg.methods,
            "target": cfg.target,
            "radii": cfg.radii,
            "agree_ratios": cfg.agree_ratios,
            "weights": weights,
            "eq_f1s": eq_wgt_f1s,
            "uneq_f1s": uneq_wgt_f1s,
        }
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    typer.run(main)
