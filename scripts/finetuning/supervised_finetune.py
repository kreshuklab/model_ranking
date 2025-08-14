import typer
from typing import Annotated


from scripts.finetuning.batch_predict_checkpoints import batch_predict_checkpoints

from model_ranking.config import copy_config
from model_ranking.supervised_training import run_supervised_training
from pytorch3dunet.unet3d.config import (
    load_config_direct,  # pyright: ignore[reportUnknownVariableType]
)
from model_ranking.dataclass import (
    MetaConfig,
    SupervisedFinetuningConfig,
)


def supervised_finetune(
    supervised_finetune_config: SupervisedFinetuningConfig, config_path: str
):
    copy_config(supervised_finetune_config, config_path)
    run_supervised_training(
        name=supervised_finetune_config.name,
        output_root=supervised_finetune_config.output_root_path,
        model_config=supervised_finetune_config.model_cfg.model,
        wandb_config=supervised_finetune_config.wandb_cfg,
        loader_config=supervised_finetune_config.loader_cfg,
        lr=supervised_finetune_config.training_cfg.lr,
        n_iterations=supervised_finetune_config.training_cfg.n_iterations,
        epochs=supervised_finetune_config.training_cfg.epochs,
        source_checkpoint=supervised_finetune_config.model_cfg.source_checkpoint,
        save_ckpt_every_kth_epoch=supervised_finetune_config.training_cfg.save_ckpt_every_kth_epoch,
        mixed_precision=supervised_finetune_config.training_cfg.mixed_precision,
        scheduler_kwargs=supervised_finetune_config.training_cfg.scheduler_kwargs,
        optimizer_kwargs=supervised_finetune_config.training_cfg.optimizer_kwargs,
    )


def main(
    config: Annotated[
        str, typer.Option(help="Path to the configuration file", exists=True)
    ],
):
    """
    Main function to run supervised fine-tuning.
    """
    cfg, _ = load_config_direct(config)
    finetune_cfg = SupervisedFinetuningConfig.model_validate(cfg["training"])
    supervised_finetune(finetune_cfg, config)

    pred_cfg = MetaConfig.model_validate(cfg["prediction"]["meta_config"])

    # Run batch predictions after fine-tuning
    batch_predict_checkpoints(
        pred_cfg, checkpoint_names=cfg["prediction"]["checkpoints"]
    )


if __name__ == "__main__":
    typer.run(main)
