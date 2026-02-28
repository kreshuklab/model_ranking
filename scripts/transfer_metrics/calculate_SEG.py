import numpy as np
from numpy.typing import NDArray
import os
from pathlib import Path
import typer
from typing import Annotated, Any, Dict, assert_never
from tqdm import tqdm

from pytorch3dunet.unet3d.config import (
    load_config_direct,  # pyright: ignore[reportUnknownVariableType]
)

from model_ranking.data_structures import SEGConfig
from model_ranking.utils import get_output_dir_paths, get_h5_dataset_size
from model_ranking.baseline_metrics._SEG import (
    get_weights,
    get_f1_scores,
    save_SEG_results,
)


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

        if cfg.task == "cells":
            paths = list(paths[0].glob("*_predictions.h5"))

            assert (
                len(paths) == 1
            ), f"Expected exactly one path for model {model}, found {len(paths)}"

        model_pred_paths[model] = paths[0]

    if cfg.task == "nuclei":
        sample_ids = sorted(
            [
                Path(f).stem
                for f in os.listdir(model_pred_paths[cfg.methods[0]])
                if f.endswith(".h5")
            ]
        )
    elif cfg.task == "cells":
        ds_shape, _ = get_h5_dataset_size(
            model_pred_paths[cfg.methods[0]], "segmentation"
        )
        sample_ids = list(range(ds_shape[0]))
    else:
        assert_never(cfg.task)

    equal_weights = np.ones((len(sample_ids), len(cfg.methods))) * (
        1 / len(cfg.methods)
    )

    weights: Dict[int, Dict[float, NDArray[Any]]] = {}
    eq_wgt_f1s: Dict[int, Dict[float, float]] = {}
    uneq_wgt_f1s: Dict[int, Dict[float, float]] = {}

    for r in tqdm(cfg.radii):
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

        save_SEG_results(cfg, weights, eq_wgt_f1s, uneq_wgt_f1s)


if __name__ == "__main__":
    typer.run(main)
