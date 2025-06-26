import os
import json
from pathlib import Path

import numpy as np
import torch
import torch_em
from tqdm import tqdm
from typing import List, Tuple, Union, Optional, Any

from torch_em.transform import (
    BoundaryTransform,
    # label,
    Compose,
    PadIfNecessary,
    get_augmentations,  # pyright: ignore[reportUnknownVariableType]
)
from torch_em.loss.dice import (
    # BCEDiceLoss,
    # DiceLoss,
    DiceLossWithLogits,
)
from torch_em.transform.raw import (
    get_default_mean_teacher_augmentations,  # pyright: ignore[reportUnknownVariableType]
)
from torch_em.data.sampler import (
    MinInstanceSampler,
    MinForegroundSampler,
    MinSemanticLabelForegroundSampler,
)
from torch_em.trainer.wandb_logger import WandbLogger


from elf.io import (  # pyright: ignore[reportMissingTypeStubs]
    open_file,  # pyright: ignore[reportUnknownVariableType]
)

from model_ranking.dataclass import Pytorch3DUnetModelConfig, WandbConfig
from model_ranking.utils import (
    is_ndarray,
)
from model_ranking.metrics import (
    DiceMetric,
)

from pytorch3dunet.unet3d.model import (
    get_model,  # pyright: ignore[reportUnknownVariableType]
)
from pytorch3dunet.unet3d.utils import (
    load_checkpoint,  # pyright: ignore[reportUnknownVariableType]
)

sampler_type = Union[
    MinInstanceSampler, MinForegroundSampler, MinSemanticLabelForegroundSampler
]


def _compute_rois(
    paths: List[str], key: str, root: Union[str, Path], patch_shape: Tuple[int, ...]
) -> List[Tuple[List[int], List[int]]]:

    def _roi(path: str, key: str):
        with open_file(path, "r") as f:  # pyright: ignore[reportUnknownVariableType]
            assert key in f, f"Could not find {key} in {path}"
            labels = f[key][:]  # pyright: ignore[reportUnknownVariableType]
            assert is_ndarray(labels), f"Labels are not a ndarray: {labels}"
            coords = np.where(labels != 0)
            start = [int(coord.min()) for coord in coords]
            stop = [int(coord.max()) + 1 for coord in coords]

            crop_shape = tuple(sto - sta for sta, sto in zip(start, stop))
            if any(csh < psh for csh, psh in zip(crop_shape, patch_shape)):
                shape_extender = [
                    (psh - csh) // 2 + 1 if psh > csh else 0
                    for csh, psh in zip(crop_shape, patch_shape)
                ]
                start = [sta - ext for sta, ext in zip(start, shape_extender)]
                stop = [sto + ext for sto, ext in zip(stop, shape_extender)]
                if any(sta < 0 for sta in start):
                    start = [max(sta, 0) for sta in start]
                    stop = [
                        sto + ext if sta == 0 else sto
                        for sta, sto, ext in zip(start, stop, shape_extender)
                    ]

                crop_shape = tuple(sto - sta for sta, sto in zip(start, stop))
            assert all(csh >= psh for csh, psh in zip(crop_shape, patch_shape))

            bb = tuple(slice(sta, sto) for sta, sto in zip(start, stop))
            labels = labels[bb]

        return start, stop

    roi_folder = os.path.join(root, "rois")
    os.makedirs(roi_folder, exist_ok=True)
    rois = []
    for path in tqdm(paths, desc="Computing rois"):
        ds_name, fname = Path(path).parts[-2], Path(path).stem
        out_path = os.path.join(
            roi_folder, f"{ds_name}_{fname}_{key.replace('/', '-')}.json"
        )

        if os.path.exists(out_path):
            with open(out_path) as f:
                this_roi = json.load(f)["roi"]
        else:
            this_roi = _roi(path, key)
            with open(out_path, "w") as f:
                json.dump({"roi": this_roi}, f)

        rois.append(this_roi)

    # for roi in rois:
    #     shape = tuple(roi[1][i] - roi[0][i] for i in range(3))
    #     print(shape)
    return rois  # pyright: ignore[reportUnknownVariableType]


def get_supervised_loader(
    data_paths: List[str],
    raw_key: str,
    label_key: str,
    patch_shape: Tuple[int, ...],
    batch_size: int,
    root: str,
    num_workers: int = 8,
    n_samples: Optional[int] = None,
    crop_to_labels: bool = False,  # NOTE: war vorher True
    add_boundary_transform: bool = False,
    label_dtype: torch.dtype = torch.float32,
    sampler: sampler_type = MinForegroundSampler(0.01),
    rois: Optional[Union[List[slice], List[Tuple[slice, ...]]]] = None,
) -> torch.utils.data.DataLoader[Any]:

    if crop_to_labels:
        print(
            "Warning: crop_to_labels is set to True, this will crop the patches to the labels. "
            + "This will overwrite rois selection if rois is not None."
        )
        rois_from_labels = _compute_rois(data_paths, label_key, root, patch_shape)
        rois = [
            tuple(slice(sta, sto) for sta, sto in zip(start, stop))
            for start, stop in rois_from_labels
        ]

    if add_boundary_transform:
        label_transform = BoundaryTransform(add_binary_target=True)
    else:
        label_transform = None
    # else:
    #    label_transform = (
    #        label.connected_components
    #    )
    transform = Compose(
        PadIfNecessary(patch_shape),
        get_augmentations(len(patch_shape)),
    )

    raw_transform = (  # pyright: ignore[reportUnknownVariableType]
        get_default_mean_teacher_augmentations()
    )

    loader = torch_em.default_segmentation_loader(  # pyright: ignore[reportUnknownVariableType]
        data_paths,
        raw_key,
        data_paths,
        label_key,
        rois=rois,
        sampler=sampler,
        batch_size=batch_size,
        patch_shape=patch_shape,
        is_seg_dataset=True,
        label_transform=label_transform,
        transform=transform,
        raw_transform=raw_transform,
        num_workers=num_workers,
        shuffle=True,
        n_samples=n_samples,
        label_dtype=label_dtype,  # pyright: ignore[reportArgumentType]
    )
    return loader  # pyright: ignore[reportUnknownVariableType]


def run_supervised_training(
    name: str,
    output_root: str,
    train_paths: List[str],
    val_paths: List[str],
    label_key: str,
    patch_shape: Tuple[int, ...],
    model_config: Pytorch3DUnetModelConfig,
    wandb_config: Optional[WandbConfig],
    raw_key: str = "raw",
    batch_size: int = 1,
    lr: float = 1e-4,
    n_iterations: Optional[int] = None,
    epochs: Optional[int] = 20,
    n_samples_train: Optional[int] = None,
    n_samples_val: Optional[int] = None,
    check: bool = False,
    loss: Optional[torch.nn.Module] = DiceLossWithLogits(),
    metric: Optional[torch.nn.Module] = DiceMetric(
        threshold=0.5, final_activation="sigmoid"
    ),
    rois_val: Optional[Union[List[slice], List[Tuple[slice, ...]]]] = None,
    rois_train: Optional[Union[List[slice], List[Tuple[slice, ...]]]] = None,
    save_ckpt_every_kth_epoch: Optional[int] = None,
    source_checkpoint: Optional[Union[str, Path]] = None,
):
    train_loader = get_supervised_loader(
        train_paths,
        raw_key,
        label_key,
        patch_shape,
        batch_size,
        output_root,
        n_samples=n_samples_train,
        rois=rois_train,
    )
    val_loader = get_supervised_loader(
        val_paths,
        raw_key,
        label_key,
        patch_shape,
        batch_size,
        output_root,
        n_samples=n_samples_val,
        rois=rois_val,
    )

    if check:
        from torch_em.util.debug import (
            check_loader,  # pyright: ignore[reportUnknownVariableType]
        )

        check_loader(train_loader, n_samples=4)
        check_loader(val_loader, n_samples=4)
        return

    model = get_model(model_config.model_dump())

    if source_checkpoint is not None:
        if Path(source_checkpoint).suffix == ".pt":
            model_key = "model_state"
        else:
            model_key = "model_state_dict"
        _ = load_checkpoint(source_checkpoint, model, model_key=model_key)

    if wandb_config is not None:
        logger_kwargs = {
            "project_name": wandb_config.project,
            "mode": wandb_config.mode,
        }
        logger = WandbLogger
    else:
        logger = None
        logger_kwargs = None

    trainer = torch_em.default_segmentation_trainer(
        name=name,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss=loss,
        metric=metric,
        learning_rate=lr,
        mixed_precision=True,
        log_image_interval=100,
        compile_model=False,
        save_root=output_root,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        logger=logger,  # pyright: ignore[reportArgumentType]
        logger_kwargs=logger_kwargs,
    )
    trainer.fit(
        iterations=n_iterations,
        save_every_kth_epoch=save_ckpt_every_kth_epoch,
        epochs=epochs,
    )
