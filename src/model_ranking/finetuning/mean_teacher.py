import torch
from typing import Optional, Union, Tuple, List, assert_never, Dict, Any
from pathlib import Path
from numpy.typing import NDArray

import torch_em.self_training as self_training

from model_ranking.configs.utils import copy_config
from model_ranking.dataclass import (
    Pytorch3DUnetModelConfig,
    UnetrModelConfig,
    pseudo_labeler_type,
    WandbConfig,
    DEFAULT_SCHEDULER_KWARGS,
    MeanTeacherConfig,
)
from model_ranking.datasets import (
    calculate_global_stats,
)
from model_ranking.logger import SelfTrainingWandbLogger
from model_ranking.metrics import DiceMetric
from .pseudo_labeling import (
    InputConsistencyPatchwisePseudoLabeler,
    ModelConsistencyPatchWisePseudoLabeler,
    ScheduledPseudoLabeler,
    DummyDirectEvalPseudoLabeler,
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

from pytorch3dunet.augment.transforms import (
    Transformer,
)
from pytorch3dunet.datasets.utils import (
    get_train_loaders,  # pyright: ignore[reportUnknownVariableType]
)


def run_mean_teacher(
    name: str,
    output_root_path: str,
    unsupervised_train_paths: List[str],
    unsupervised_val_paths: List[str],
    patch_shape: Tuple[int, ...],
    pseudo_labeler_config: pseudo_labeler_type,
    model_config: Union[Pytorch3DUnetModelConfig, UnetrModelConfig],
    wandb_config: Optional[WandbConfig],
    source_checkpoint: Optional[Union[str, Path]] = None,
    supervised_loader_config: Optional[Dict[str, Any]] = None,
    raw_key: str = "raw",
    label_key: Optional[str] = None,
    batch_size: int = 1,
    num_workers: int = 8,
    lr: float = 1e-4,
    n_iterations: Optional[int] = None,
    epochs: Optional[int] = 10,
    n_samples_train: Optional[int] = None,
    n_samples_val: Optional[int] = None,
    save_ckpt_every_kth_epoch: Optional[int] = None,
    roi_unsupervised_train: Optional[Union[slice, Tuple[slice, ...]]] = None,
    roi_unsupervised_val: Optional[Union[slice, Tuple[slice, ...]]] = None,
    scheduler_kwargs: Dict[str, Any] = DEFAULT_SCHEDULER_KWARGS,
    optimizer_kwargs: Dict[str, Any] = {},
    mixed_precision: bool = True,
    global_normalisation: bool = False,
    norm01: bool = False,
):
    assert (n_iterations is None) != (
        epochs is None
    ), "Specify exactly one of n_iterations or epochs (not both, not neither)"

    model = get_model(model_config.model_dump())
    if source_checkpoint is None:
        # training from scratch only makes sense if we have supervised training data
        # that's why we have the assertion here.
        assert supervised_loader_config is not None
        print("Mean teacher training from scratch")
        reinit_teacher = True
    else:
        print("Mean teacher training initialized from source model:", source_checkpoint)
        if Path(source_checkpoint).suffix == ".pt":
            model_key = "model_state"
        else:
            model_key = "model_state_dict"
        _ = load_checkpoint(source_checkpoint, model, model_key=model_key)
        reinit_teacher = False

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, **optimizer_kwargs)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, **scheduler_kwargs
    )

    if pseudo_labeler_config.activation is not None:
        if pseudo_labeler_config.activation == "softmax":
            activation = torch.nn.Softmax(dim=1)
        elif pseudo_labeler_config.activation == "sigmoid":
            activation = torch.nn.Sigmoid()
        else:
            raise ValueError(
                f"Unknown activation: {pseudo_labeler_config.activation}. "
                + "Supported are 'softmax' and 'sigmoid'."
            )
    else:
        activation = None

    # Get the consistency metric
    if (pseudo_labeler_config.name == "input_consistency") or (
        pseudo_labeler_config.name == "model_consistency"
    ):
        consis_cfg = pseudo_labeler_config.consistency_metric
        if consis_cfg.name == "AdaptedRandError":
            consistency_metric = consis_cfg.initialise_metric(incomplete_gt=False)
        else:
            consistency_metric = consis_cfg.initialise_metric()

        # self training functionality
        if pseudo_labeler_config.name == "input_consistency":

            pseudo_labeler = InputConsistencyPatchwisePseudoLabeler(
                transformer=Transformer(
                    pseudo_labeler_config.transformer_cfg,
                    pseudo_labeler_config.stats_cfg,
                ),
                consistency_metric=consistency_metric,
                mask_threshold=consis_cfg.mask_threshold,
                consistency_threshold=pseudo_labeler_config.consistency_threshold,
                seg_params=pseudo_labeler_config.seg_params,
                activation=activation,
            )

        else:
            pseudo_labeler = ModelConsistencyPatchWisePseudoLabeler(
                perturbed_model_config=pseudo_labeler_config.perturbed_model_config,
                consistency_metric=consistency_metric,
                mask_threshold=consis_cfg.mask_threshold,
                consistency_threshold=pseudo_labeler_config.consistency_threshold,
                seg_params=pseudo_labeler_config.seg_params,
                activation=activation,
            )

    elif pseudo_labeler_config.name == "default_pseudo_labeler":
        pseudo_labeler = self_training.DefaultPseudoLabeler(
            activation=activation,
            confidence_threshold=pseudo_labeler_config.confidence_threshold,
            threshold_from_both_sides=pseudo_labeler_config.threshold_from_both_sides,
        )
    elif pseudo_labeler_config.name == "scheduled_pseudo_labeler":
        pseudo_labeler = ScheduledPseudoLabeler(
            activation=activation,
            confidence_threshold=pseudo_labeler_config.confidence_threshold,
            threshold_from_both_sides=pseudo_labeler_config.threshold_from_both_sides,
            mode=pseudo_labeler_config.mode,
            factor=pseudo_labeler_config.factor,
            patience=pseudo_labeler_config.patience,
            threshold=pseudo_labeler_config.threshold,
            threshold_mode=pseudo_labeler_config.threshold_mode,
            min_ct=pseudo_labeler_config.min_ct,
            eps=pseudo_labeler_config.eps,
            verbose=pseudo_labeler_config.verbose,
        )

    elif pseudo_labeler_config.name == "direct_eval_pseudo_labeler":
        pseudo_labeler = DummyDirectEvalPseudoLabeler(
            score_threshold=pseudo_labeler_config.score_threshold,
            activation=activation,
        )

    else:
        assert_never(pseudo_labeler_config.name)

    loss = self_training.DefaultSelfTrainingLoss(activation=torch.nn.Sigmoid())
    loss_and_metric = self_training.DefaultSelfTrainingLossAndMetric(
        metric=DiceMetric(),
        activation=torch.nn.Sigmoid(),
    )

    if global_normalisation:
        raw_train: List[NDArray[Any]] = []
        raw_val: List[NDArray[Any]] = []
        for train_path, val_path in zip(
            unsupervised_train_paths, unsupervised_val_paths
        ):
            assert (Path(train_path).suffix == ".h5") and (
                Path(val_path).suffix == ".h5"
            ), "Global normalisation only works with h5 files."
            raw_train.append(load_h5(train_path, raw_key))
            raw_val.append(load_h5(val_path, raw_key))
        raw_stats = calculate_global_stats(raw_train)
        val_stats = calculate_global_stats(raw_val)
    else:
        raw_stats = None
        val_stats = None

    if isinstance(pseudo_labeler, DummyDirectEvalPseudoLabeler):
        # For the dummy pseudo labeler, we use the DummySelfTrainingLoader
        assert (
            label_key is not None
        ), "label_key must be provided for DummySelfTrainingLoader"
        unsupervised_train_loader = get_DummySelfTraining_loader(
            unsupervised_train_paths,
            raw_key,
            label_key,
            patch_shape,
            batch_size,
            n_samples=n_samples_train,
            roi=roi_unsupervised_train,
            global_stats=raw_stats,
            norm01=norm01,
        )
        unsupervised_val_loader = get_DummySelfTraining_loader(
            unsupervised_val_paths,
            raw_key,
            label_key,
            patch_shape,
            batch_size,
            n_samples=n_samples_val,
            roi=roi_unsupervised_val,
            global_stats=val_stats,
            norm01=norm01,
        )

    else:
        print("Get unsup loaders")
        unsupervised_train_loader = get_unsupervised_loader(
            unsupervised_train_paths,
            raw_key,
            patch_shape,
            batch_size,
            num_workers=num_workers,
            n_samples=n_samples_train,
            roi=roi_unsupervised_train,
            global_stats=raw_stats,
            norm01=norm01,
        )
        unsupervised_val_loader = get_unsupervised_loader(
            unsupervised_val_paths,
            raw_key,
            patch_shape,
            batch_size,
            num_workers=num_workers,
            n_samples=n_samples_val,
            roi=roi_unsupervised_val,
            global_stats=val_stats,
            norm01=norm01,
        )

    if supervised_loader_config is not None:
        print("Get supervised loaders with config")
        supervised_loaders = (  # pyright: ignore[reportUnknownVariableType]
            get_train_loaders(supervised_loader_config)
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
    if wandb_config is not None:
        logger_kwargs = {
            "project_name": wandb_config.project,
            "mode": wandb_config.mode,
        }
        logger = SelfTrainingWandbLogger
    else:
        logger = None
        logger_kwargs = None
    trainer = self_training.MeanTeacherTrainer(
        name=name,
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
        mixed_precision=mixed_precision,
        log_image_interval=100,
        compile_model=False,
        device=device,
        reinit_teacher=reinit_teacher,
        save_root=output_root_path,
    )
    trainer.fit(
        iterations=n_iterations,
        save_every_kth_epoch=save_ckpt_every_kth_epoch,
        epochs=epochs,
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
        name=mean_teacher_config.name,
        output_root_path=mean_teacher_config.output_root_path,
        unsupervised_train_paths=mean_teacher_config.data_cfg.unsupervised_train_paths,
        unsupervised_val_paths=mean_teacher_config.data_cfg.unsupervised_val_paths,
        patch_shape=mean_teacher_config.data_cfg.patch_shape,
        pseudo_labeler_config=mean_teacher_config.pseudo_labeler_cfg,
        model_config=mean_teacher_config.model_cfg.model,
        supervised_loader_config=mean_teacher_config.supervised_loader_cfg,
        source_checkpoint=mean_teacher_config.model_cfg.source_checkpoint,
        raw_key=mean_teacher_config.data_cfg.raw_key,
        label_key=mean_teacher_config.data_cfg.label_key,
        batch_size=mean_teacher_config.data_cfg.batch_size,
        lr=mean_teacher_config.training_cfg.lr,
        n_iterations=mean_teacher_config.training_cfg.n_iterations,
        epochs=mean_teacher_config.training_cfg.epochs,
        n_samples_train=mean_teacher_config.data_cfg.n_samples_train,
        n_samples_val=mean_teacher_config.data_cfg.n_samples_val,
        save_ckpt_every_kth_epoch=mean_teacher_config.training_cfg.save_ckpt_every_kth_epoch,
        wandb_config=mean_teacher_config.wandb_cfg,
        roi_unsupervised_train=roi_unsupervised_train,
        roi_unsupervised_val=roi_unsupervised_val,
        mixed_precision=mean_teacher_config.training_cfg.mixed_precision,
        global_normalisation=mean_teacher_config.data_cfg.global_normalization,
        norm01=mean_teacher_config.data_cfg.norm01,
    )
