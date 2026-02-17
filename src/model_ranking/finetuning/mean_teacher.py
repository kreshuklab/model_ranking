import torch
from typing import Optional, Union, Tuple, List, Dict, Any
from pathlib import Path
from numpy.typing import NDArray

import torch_em.self_training as self_training

from model_ranking.configs.utils import copy_config
from model_ranking.data_structures import (
    DEFAULT_SCHEDULER_KWARGS,
    MeanTeacherConfig,
)
from model_ranking.datasets import (
    calculate_global_stats,
)
from model_ranking.logger import SelfTrainingWandbLogger
from model_ranking.metrics import DiceMetric
from .pseudo_labeling import (
    DummyDirectEvalPseudoLabeler,
    get_pseudo_labeler,
)
from .self_training import (
    get_unsupervised_loader,
    get_DummySelfTraining_loader,
)

from pytorch3dunet.unet3d.model import (
    get_model,  # pyright: ignore[reportUnknownVariableType]
)
from pytorch3dunet.unet3d.utils import (
    load_checkpoint,  # pyright: ignore[reportUnknownVariableType]
)
from model_ranking.utils import load_h5, get_roi_slice

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

    model = get_model(model_cfg.model_dump())
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

    if data_cfg.normalisation.global_normalisation:
        raw_train: List[NDArray[Any]] = []
        raw_val: List[NDArray[Any]] = []
        for train_path, val_path in zip(
            data_cfg.unsupervised_train_paths, data_cfg.unsupervised_val_paths
        ):
            assert (Path(train_path).suffix == ".h5") and (
                Path(val_path).suffix == ".h5"
            ), "Global normalisation only works with h5 files."
            raw_train.append(load_h5(train_path, data_cfg.raw_key))
            raw_val.append(load_h5(val_path, data_cfg.raw_key))
        raw_stats = calculate_global_stats(raw_train)
        val_stats = calculate_global_stats(raw_val)
    else:
        raw_stats = None
        val_stats = None

    if isinstance(pseudo_labeler, DummyDirectEvalPseudoLabeler):
        # For the dummy pseudo labeler, we use the DummySelfTrainingLoader
        assert (
            data_cfg.label_key is not None
        ), "label_key must be provided for DummySelfTrainingLoader"
        unsupervised_train_loader = get_DummySelfTraining_loader(
            data_cfg.unsupervised_train_paths,
            data_cfg.raw_key,
            data_cfg.label_key,
            data_cfg.patch_shape,
            data_cfg.batch_size,
            n_samples=data_cfg.n_samples_train,
            roi=roi_unsupervised_train,
            global_stats=raw_stats,
            norm01=data_cfg.normalisation.norm01,
        )
        unsupervised_val_loader = get_DummySelfTraining_loader(
            data_cfg.unsupervised_val_paths,
            data_cfg.raw_key,
            data_cfg.label_key,
            data_cfg.patch_shape,
            data_cfg.batch_size,
            n_samples=data_cfg.n_samples_val,
            roi=roi_unsupervised_val,
            global_stats=val_stats,
            norm01=data_cfg.normalisation.norm01,
        )

    else:
        print("Get unsup loaders")
        unsupervised_train_loader = get_unsupervised_loader(
            data_cfg.unsupervised_train_paths,
            data_cfg.raw_key,
            data_cfg.patch_shape,
            data_cfg.batch_size,
            num_workers=data_cfg.num_workers,
            n_samples=data_cfg.n_samples_train,
            roi=roi_unsupervised_train,
            global_stats=raw_stats,
            norm01=data_cfg.normalisation.norm01,
        )
        unsupervised_val_loader = get_unsupervised_loader(
            data_cfg.unsupervised_val_paths,
            data_cfg.raw_key,
            data_cfg.patch_shape,
            data_cfg.batch_size,
            num_workers=data_cfg.num_workers,
            n_samples=data_cfg.n_samples_val,
            roi=roi_unsupervised_val,
            global_stats=val_stats,
            norm01=data_cfg.normalisation.norm01,
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
