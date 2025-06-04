from typing import Annotated
import typer

from pytorch3dunet.unet3d.config import (
    load_config_direct,  # pyright: ignore[reportUnknownVariableType]
)
from mean_teacher_finetune import self_training_mean_teacher
from batch_predict_checkpoints import batch_predict_checkpoints
from model_ranking.dataclass import (
    MetaConfig,
    MeanTeacherConfig,
)


def main(
    config: Annotated[
        str, typer.Option(help="Path to the configuration file", exists=True)
    ],
):
    """
    Main function to run mean teacher fine-tuning and batch predictions for multiple checkpoints.
    """
    cfg, _ = load_config_direct(config)
    train_cfg = MeanTeacherConfig.model_validate(cfg["training"])
    self_training_mean_teacher(train_cfg, config)

    pred_cfg = MetaConfig.model_validate(cfg["prediction"]["meta_config"])

    # Run batch predictions after fine-tuning
    batch_predict_checkpoints(
        pred_cfg, checkpoint_names=cfg["prediction"]["checkpoints"]
    )


if __name__ == "__main__":
    typer.run(main)
