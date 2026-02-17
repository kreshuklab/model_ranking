import torch
from typing import Optional, Union, Tuple, Dict, Any
from pathlib import Path

import torch_em.self_training as self_training

from model_ranking.configs.utils import copy_config
from model_ranking.data_structures import (
    DEFAULT_SCHEDULER_KWARGS,
    MeanTeacherConfig,
)
from model_ranking.logger import SelfTrainingWandbLogger
from model_ranking.metrics import DiceMetric
from model_ranking.utils import get_roi_slice
from .pseudo_labeling import (
    get_pseudo_labeler,
)
from .self_training import (
    get_MT_unsupervised_loaders,
)

from pytorch3dunet.unet3d.model import (
    get_model,  # pyright: ignore[reportUnknownVariableType]
)
from pytorch3dunet.unet3d.utils import (
    load_checkpoint,  # pyright: ignore[reportUnknownVariableType]
)
from pytorch3dunet.datasets.utils import (
    get_train_loaders,  # pyright: ignore[reportUnknownVariableType]
)


def run_mean_teacher(
    config: MeanTeacherConfig,
    roi_unsupervised_train: Optional[Union[slice, Tuple[slice, ...]]] = None,
    roi_unsupervised_val: Optional[Union[slice, Tuple[slice, ...]]] = None,
    scheduler_kwargs: Dict[str, Any] = DEFAULT_SCHEDULER_KWARGS,
    optimizer_kwargs: Dict[str, Any] = {},
):
    trn_cfg = config.training_cfg
    data_cfg = config.data_cfg
    sup_data_cfg = config.supervised_loader_cfg
    psd_cfg = config.pseudo_labeler_cfg
    log_cfg = config.wandb_cfg
    model_cfg = config.model_cfg

    assert (trn_cfg.n_iterations is None) != (
        trn_cfg.epochs is None
    ), "Specify exactly one of n_iterations or epochs (not both, not neither)"

    model = get_model(model_cfg.model.model_dump())
    if model_cfg.source_checkpoint is None:
        # training from scratch only makes sense if we have supervised training data
        # that's why we have the assertion here.
        assert sup_data_cfg is not None
        print("Mean teacher training from scratch")
        reinit_teacher = True
    else:
        print(
            "Mean teacher training initialized from source model:",
            model_cfg.source_checkpoint,
        )
        if Path(model_cfg.source_checkpoint).suffix == ".pt":
            model_key = "model_state"
        else:
            model_key = "model_state_dict"
        _ = load_checkpoint(model_cfg.source_checkpoint, model, model_key=model_key)
        reinit_teacher = False

    optimizer = torch.optim.Adam(model.parameters(), lr=trn_cfg.lr, **optimizer_kwargs)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, **scheduler_kwargs
    )

    pseudo_labeler = get_pseudo_labeler(psd_cfg)

    loss = self_training.DefaultSelfTrainingLoss(activation=torch.nn.Sigmoid())
    loss_and_metric = self_training.DefaultSelfTrainingLossAndMetric(
        metric=DiceMetric(),
        activation=torch.nn.Sigmoid(),
    )

    unsupervised_train_loader, unsupervised_val_loader = get_MT_unsupervised_loaders(
        data_cfg,
        roi_unsupervised_train=roi_unsupervised_train,
        roi_unsupervised_val=roi_unsupervised_val,
        psuedo_labeler_name=psd_cfg.name,
    )

    if sup_data_cfg is not None:
        print("Get supervised loaders with config")
        supervised_loaders = (  # pyright: ignore[reportUnknownVariableType]
            get_train_loaders(sup_data_cfg)
        )
        supervised_train_loader = (  # pyright: ignore[reportUnknownVariableType]
            supervised_loaders["train"]
        )
        supervised_val_loader = (  # pyright: ignore[reportUnknownVariableType]
            supervised_loaders["val"]
        )

    else:
        supervised_train_loader = None
        supervised_val_loader = None

    print("Lift off!")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if log_cfg is not None:
        logger_kwargs = {
            "project_name": log_cfg.project,
            "mode": log_cfg.mode,
        }
        logger = SelfTrainingWandbLogger
    else:
        logger = None
        logger_kwargs = None
    trainer = self_training.MeanTeacherTrainer(
        name=config.name,
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        pseudo_labeler=pseudo_labeler,
        unsupervised_loss=loss,  # pyright: ignore[reportArgumentType]
        unsupervised_loss_and_metric=loss_and_metric,
        supervised_train_loader=supervised_train_loader,
        unsupervised_train_loader=unsupervised_train_loader,
        supervised_val_loader=supervised_val_loader,
        unsupervised_val_loader=unsupervised_val_loader,
        supervised_loss=loss,
        supervised_loss_and_metric=loss_and_metric,
        logger=logger,  # pyright: ignore[reportArgumentType]
        logger_kwargs=logger_kwargs,
        mixed_precision=trn_cfg.mixed_precision,
        log_image_interval=100,
        compile_model=False,
        device=device,
        reinit_teacher=reinit_teacher,
        save_root=config.output_root_path,
    )
    trainer.fit(
        iterations=trn_cfg.n_iterations,
        save_every_kth_epoch=trn_cfg.save_ckpt_every_kth_epoch,
        epochs=trn_cfg.epochs,
    )


def self_training_mean_teacher(
    mean_teacher_config: MeanTeacherConfig, config_path: str
):
    copy_config(mean_teacher_config, config_path)
    if mean_teacher_config.data_cfg.roi_unsupervised_train is not None:
        roi_unsupervised_train = get_roi_slice(
            mean_teacher_config.data_cfg.roi_unsupervised_train
        )
    else:
        roi_unsupervised_train = None

    if mean_teacher_config.data_cfg.roi_unsupervised_val is not None:
        roi_unsupervised_val = get_roi_slice(
            mean_teacher_config.data_cfg.roi_unsupervised_val
        )
    else:
        roi_unsupervised_val = None

    run_mean_teacher(
        config=mean_teacher_config,
        roi_unsupervised_train=roi_unsupervised_train,
        roi_unsupervised_val=roi_unsupervised_val,
    )
