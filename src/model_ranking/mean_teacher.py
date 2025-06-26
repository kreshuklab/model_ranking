import torch
from typing import Optional, Union, Tuple, List, assert_never
from pathlib import Path

import torch_em.self_training as self_training
from model_ranking.dataclass import (
    Pytorch3DUnetModelConfig,
    pseudo_labeler_type,
    WandbConfig,
)
from model_ranking.logger import SelfTrainingWandbLogger
from model_ranking.pseudo_labeling import (
    InputConsistencyPatchwisePseudoLabeler,
    ModelConsistencyPatchWisePseudoLabeler,
    ScheduledPseudoLabeler,
    DummyDirectEvalPseudoLabeler,
)
from model_ranking.self_training import (
    get_unsupervised_loader,
    get_DummySelfTraining_loader,
)
from model_ranking.supervised_training import get_supervised_loader
from model_ranking.metrics import DiceMetric

from pytorch3dunet.unet3d.model import (
    get_model,  # pyright: ignore[reportUnknownVariableType]
)
from pytorch3dunet.unet3d.utils import (
    load_checkpoint,  # pyright: ignore[reportUnknownVariableType]
)
from pytorch3dunet.augment.transforms import (
    Transformer,
)


def run_mean_teacher(
    name: str,
    output_root_path: str,
    unsupervised_train_paths: List[str],
    unsupervised_val_paths: List[str],
    patch_shape: Tuple[int, ...],
    pseudo_labeler_config: pseudo_labeler_type,
    model_config: Pytorch3DUnetModelConfig,
    wandb_config: Optional[WandbConfig],
    source_checkpoint: Optional[Union[str, Path]] = None,
    supervised_train_paths: Optional[List[str]] = None,
    supervised_val_paths: Optional[List[str]] = None,
    raw_key: str = "raw",
    raw_key_supervised: Optional[str] = "raw",
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
    roi_supervised_train: Optional[Union[List[slice], List[Tuple[slice, ...]]]] = None,
    roi_supervised_val: Optional[Union[List[slice], List[Tuple[slice, ...]]]] = None,
):
    assert (n_iterations is None) != (
        epochs is None
    ), "Specify exactly one of n_iterations or epochs (not both, not neither)"

    assert (supervised_train_paths is None) == (supervised_val_paths is None)

    model = get_model(model_config.model_dump())
    if source_checkpoint is None:
        # training from scratch only makes sense if we have supervised training data
        # that's why we have the assertion here.
        assert supervised_train_paths is not None
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

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
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
        )

    else:
        assert_never(pseudo_labeler_config.name)

    loss = self_training.DefaultSelfTrainingLoss(activation=torch.nn.Sigmoid())
    loss_and_metric = self_training.DefaultSelfTrainingLossAndMetric(
        metric=DiceMetric(threshold=0.5),
        activation=torch.nn.Sigmoid(),
    )

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
        )
        unsupervised_val_loader = get_DummySelfTraining_loader(
            unsupervised_val_paths,
            raw_key,
            label_key,
            patch_shape,
            batch_size,
            n_samples=n_samples_val,
            roi=roi_unsupervised_val,
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
        )
        unsupervised_val_loader = get_unsupervised_loader(
            unsupervised_val_paths,
            raw_key,
            patch_shape,
            batch_size,
            num_workers=num_workers,
            n_samples=n_samples_val,
            roi=roi_unsupervised_val,
        )

    if supervised_train_paths is not None:
        print("Get sup loaders")
        assert (label_key is not None) and (
            raw_key_supervised is not None
        ), f"label_key: {label_key}, raw_key_supervised: {raw_key_supervised}"
        supervised_train_loader = get_supervised_loader(
            supervised_train_paths,
            raw_key_supervised,
            label_key,
            patch_shape,
            batch_size,
            output_root_path,
            num_workers=num_workers,
            n_samples=n_samples_train,
            rois=roi_supervised_train,
        )
        assert supervised_val_paths is not None
        supervised_val_loader = get_supervised_loader(
            supervised_val_paths,
            raw_key_supervised,
            label_key,
            patch_shape,
            batch_size,
            output_root_path,
            num_workers=num_workers,
            n_samples=n_samples_val,
            rois=roi_supervised_val,
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
        mixed_precision=True,
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
