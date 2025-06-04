import typer
from typing import Annotated

from model_ranking.dataclass import MeanTeacherConfig
from model_ranking.mean_teacher import run_mean_teacher
from pytorch3dunet.unet3d.config import (
    load_config_direct,  # pyright: ignore[reportUnknownVariableType]
)


def main(
    config: Annotated[str, typer.Option(help="Path to config file", exists=True)],
):
    cfg, _ = load_config_direct(config)
    mt_cfg = MeanTeacherConfig.model_validate(cfg)

    run_mean_teacher(
        name=mt_cfg.name,
        output_root_path=mt_cfg.output_root_path,
        unsupervised_train_paths=mt_cfg.data_cfg.unsupervised_train_paths,
        unsupervised_val_paths=mt_cfg.data_cfg.unsupervised_val_paths,
        patch_shape=mt_cfg.data_cfg.patch_shape,
        pseudo_labeler_config=mt_cfg.pseudo_labeler_cfg,
        model_config=mt_cfg.model_cfg.model,
        source_checkpoint=mt_cfg.model_cfg.source_checkpoint,
        supervised_train_paths=mt_cfg.data_cfg.supervised_train_paths,
        supervised_val_paths=mt_cfg.data_cfg.supervised_val_paths,
        raw_key=mt_cfg.data_cfg.raw_key,
        raw_key_supervised=mt_cfg.data_cfg.raw_key_supervised,
        label_key=mt_cfg.data_cfg.label_key,
        batch_size=mt_cfg.data_cfg.batch_size,
        lr=mt_cfg.training_cfg.lr,
        n_iterations=mt_cfg.training_cfg.n_iterations,
        epochs=mt_cfg.training_cfg.epochs,
        n_samples_train=mt_cfg.data_cfg.n_samples_train,
        n_samples_val=mt_cfg.data_cfg.n_samples_val,
        save_ckpt_every_kth_epoch=mt_cfg.training_cfg.save_ckpt_every_kth_epoch,
        wandb_config=mt_cfg.wandb_cfg,
    )


if __name__ == "__main__":
    typer.run(main)
