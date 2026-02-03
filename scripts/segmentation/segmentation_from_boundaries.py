import typer
from typing import Annotated

from model_ranking.data_structures import (
    CalculateSegmentationConfig,
)
from model_ranking.predict import (
    calculate_segmentation,
    get_pred_paths,
)
from model_ranking.utils import (
    load_h5,
    save_h5,
)
from pytorch3dunet.unet3d.config import (
    load_config_direct,  # pyright: ignore[reportUnknownVariableType]
)


def main(
    config: Annotated[str, typer.Option(help="Path to Meta config file", exists=True)],
):
    cfg_values, _ = load_config_direct(config)
    cfg = CalculateSegmentationConfig.model_validate(cfg_values)

    pred_paths = get_pred_paths(cfg.pred_base_path, cfg.models)

    for pred_path in pred_paths:
        print(f"Processing {pred_path}")
        preds = load_h5(pred_path, cfg.pred_key)
        segs = calculate_segmentation(
            preds,
            min_size=cfg.min_size,
            zero_largest_instance=cfg.zero_largest_instance,
        )

        save_h5(pred_path, cfg.output_key, segs, overwrite=cfg.overwrite_output)


if __name__ == "__main__":
    typer.run(main)
