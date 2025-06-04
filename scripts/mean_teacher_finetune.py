import typer
from typing import Annotated

from model_ranking.config import copy_config
from model_ranking.dataclass import MeanTeacherConfig
from model_ranking.mean_teacher import run_mean_teacher
from pytorch3dunet.unet3d.config import (
    load_config_direct,  # pyright: ignore[reportUnknownVariableType]
)


def self_training_mean_teacher(
    mean_teacher_config: MeanTeacherConfig, config_path: str
):
    run_mean_teacher(
        name=mean_teacher_config.name,
        output_root_path=mean_teacher_config.output_root_path,
        unsupervised_train_paths=mean_teacher_config.data_cfg.unsupervised_train_paths,
        unsupervised_val_paths=mean_teacher_config.data_cfg.unsupervised_val_paths,
        patch_shape=mean_teacher_config.data_cfg.patch_shape,
        pseudo_labeler_config=mean_teacher_config.pseudo_labeler_cfg,
        model_config=mean_teacher_config.model_cfg.model,
        source_checkpoint=mean_teacher_config.model_cfg.source_checkpoint,
        supervised_train_paths=mean_teacher_config.data_cfg.supervised_train_paths,
        supervised_val_paths=mean_teacher_config.data_cfg.supervised_val_paths,
        raw_key=mean_teacher_config.data_cfg.raw_key,
        raw_key_supervised=mean_teacher_config.data_cfg.raw_key_supervised,
        label_key=mean_teacher_config.data_cfg.label_key,
        batch_size=mean_teacher_config.data_cfg.batch_size,
        lr=mean_teacher_config.training_cfg.lr,
        n_iterations=mean_teacher_config.training_cfg.n_iterations,
        epochs=mean_teacher_config.training_cfg.epochs,
        n_samples_train=mean_teacher_config.data_cfg.n_samples_train,
        n_samples_val=mean_teacher_config.data_cfg.n_samples_val,
        save_ckpt_every_kth_epoch=mean_teacher_config.training_cfg.save_ckpt_every_kth_epoch,
        wandb_config=mean_teacher_config.wandb_cfg,
    )
    copy_config(mean_teacher_config, config_path)


def main(
    config: Annotated[str, typer.Option(help="Path to config file", exists=True)],
):
    cfg, _ = load_config_direct(config)
    mt_cfg = MeanTeacherConfig.model_validate(cfg)

    self_training_mean_teacher(mt_cfg, config)


if __name__ == "__main__":
    typer.run(main)
