import torch
from typing import Optional, Union, Tuple, List
from pathlib import Path

import torch_em.self_training as self_training  # pyright: ignore[reportMissingTypeStubs]


from model_ranking.dataclass import (
    Pytorch3DUnetModelConfig,
    pseudo_labeler_type,
)
from model_ranking.pseudo_labeling import (
    InputConsistencyPatchwisePseudoLabeler,
    ModelConsistencyPatchWisePseudoLabeler,
    consistency_metrics,
)
from model_ranking.self_training import get_unsupervised_loader
from model_ranking.supervised_training import get_supervised_loader

from pytorch3dunet.unet3d.model import (
    get_model,  # pyright: ignore[reportUnknownVariableType]
)
from pytorch3dunet.unet3d.utils import (
    load_checkpoint,  # pyright: ignore[reportUnknownVariableType]
)
from pytorch3dunet.augment.transforms import (
    Transformer,
)


def mean_teacher_adaptation(
    name: str,
    output_root_path: str,
    unsupervised_train_paths: List[str],
    unsupervised_val_paths: List[str],
    patch_shape: Tuple[int, ...],
    pseudo_labeler_config: pseudo_labeler_type,
    consistency_metric: consistency_metrics,
    model_config: Pytorch3DUnetModelConfig,
    source_checkpoint: Optional[Union[str, Path]] = None,
    supervised_train_paths: Optional[List[str]] = None,
    supervised_val_paths: Optional[List[str]] = None,
    raw_key: str = "raw",
    raw_key_supervised: Optional[str] = "raw",
    label_key: Optional[str] = None,
    batch_size: int = 1,
    lr: float = 1e-4,
    n_iterations: int = int(1e4),
    n_samples_train: Optional[int] = None,
    n_samples_val: Optional[int] = None,
):

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
        _ = load_checkpoint(source_checkpoint, model)
        reinit_teacher = False

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # self training functionality
    if pseudo_labeler_config.name == "input_consistency":

        pseudo_labeler = InputConsistencyPatchwisePseudoLabeler(
            transformer=Transformer(
                pseudo_labeler_config.transformer_cfg,
                pseudo_labeler_config.stats_cfg,
            ),
            consistency_metric=consistency_metric,
            foreground_threshold=pseudo_labeler_config.foreground_threshold,
            consistency_threshold=pseudo_labeler_config.consistency_threshold,
            seg_params=pseudo_labeler_config.seg_params,
        )

    else:
        pseudo_labeler = ModelConsistencyPatchWisePseudoLabeler(
            perturbed_model_config=pseudo_labeler_config.perturbed_model_config,
            consistency_metric=consistency_metric,
            foreground_threshold=pseudo_labeler_config.foreground_threshold,
            consistency_threshold=pseudo_labeler_config.consistency_threshold,
            seg_params=pseudo_labeler_config.seg_params,
        )

    loss = self_training.DefaultSelfTrainingLoss()
    loss_and_metric = self_training.DefaultSelfTrainingLossAndMetric()

    print("Get unsup loaders")
    unsupervised_train_loader = get_unsupervised_loader(
        unsupervised_train_paths,
        raw_key,
        patch_shape,
        batch_size,
        n_samples=n_samples_train,
    )
    unsupervised_val_loader = get_unsupervised_loader(
        unsupervised_val_paths,
        raw_key,
        patch_shape,
        batch_size,
        n_samples=n_samples_val,
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
            n_samples=n_samples_train,
        )
        assert supervised_val_paths is not None
        supervised_val_loader = get_supervised_loader(
            supervised_val_paths,
            raw_key_supervised,
            label_key,
            patch_shape,
            batch_size,
            output_root_path,
            n_samples=n_samples_val,
        )
    else:
        supervised_train_loader = None
        supervised_val_loader = None

    print("Lift off!")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
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
        logger=self_training.SelfTrainingTensorboardLogger,
        mixed_precision=True,
        log_image_interval=100,
        compile_model=False,
        device=device,
        reinit_teacher=reinit_teacher,
        save_root=output_root_path,
    )
    trainer.fit(n_iterations)
