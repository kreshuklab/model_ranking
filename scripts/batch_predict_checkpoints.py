from typing import Annotated, List
import typer

from pytorch3dunet.unet3d.config import (
    load_config_direct,  # pyright: ignore[reportUnknownVariableType]
)
from model_ranking.dataclass import (
    MetaConfig,
)
from model_ranking.yaml_generators import generate_run_yamls
from predict_eval import predict_eval


def batch_predict_checkpoints(
    meta_config: MetaConfig,
    checkpoint_names: List[str] = ["best"],  # Comma-separated list of checkpoints
):

    # Run predictions using the specified configuration and checkpoints.

    for checkpoint in checkpoint_names:
        # Update the checkpoint in the configuration
        for source_model in meta_config.source_models:
            source_model.checkpoint_name = checkpoint
        meta_config.output_settings.output_folder = checkpoint
        pred_yaml_paths = generate_run_yamls(meta_config.model_dump())
        for transfer in pred_yaml_paths.keys():
            for pred_yaml_path in pred_yaml_paths[transfer]:
                pred_cfg, _ = load_config_direct(pred_yaml_path)
                print(f"Running prediction with config: {pred_yaml_path}")
                predict_eval(pred_cfg)


def main(
    config: Annotated[str, typer.Option(help="Path to the configuration file")],
    checkpoints: Annotated[
        str, typer.Option(help="Comma-separated list of checkpoints")
    ] = "best",
):
    """
    Main function to run predictions for multiple checkpoints.
    """
    checkpoints_list = [ckpt.strip() for ckpt in checkpoints.split(",")]
    meta_cfg, _ = load_config_direct(config)
    meta_cfg = MetaConfig.model_validate(meta_cfg)
    batch_predict_checkpoints(meta_cfg, checkpoints_list)


if __name__ == "__main__":
    typer.run(main)
