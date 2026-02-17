from functools import partial
from pathlib import Path
import torch
from typing import Optional, Tuple, Union, List, Any, Dict
from torch.utils.data import ConcatDataset

from model_ranking.augmentations import (
    normalize_specify_range,
    weak_augmentations,
)
from model_ranking.datasets import DummySelfTrainingDataset, calculate_global_stats
from model_ranking.data_structures import SelfTrainingDataConfig, psuedo_labeler_names
from model_ranking.utils import load_h5

from torch_em.data import RawDataset
from torch_em.segmentation import (
    get_data_loader,  # pyright: ignore[reportUnknownVariableType]
)
from torch_em.transform import (
    get_raw_transform,  # pyright: ignore[reportUnknownVariableType]
    get_augmentations,  # pyright: ignore[reportUnknownVariableType]
)
from torch_em.transform.raw import (
    RawTransform,
)

from pytorch3dunet.datasets.hdf5 import (
    traverse_h5_paths,  # pyright: ignore[reportUnknownVariableType]
)


def _build_raw_transform(
    path: str,
    global_stats: Optional[Dict[str, Dict[str, Any]]],
    norm01: Optional[bool],
) -> RawTransform:
    if global_stats is not None:
        if global_stats[path].get("pmin") is not None:
            minval = global_stats[path]["pmin"]
            maxval = global_stats[path]["pmax"]
        else:
            minval = global_stats[path]["min"]
            maxval = global_stats[path]["max"]
        norm = partial(
            normalize_specify_range,
            minval=minval,
            maxval=maxval,
            norm01=norm01,
        )
    else:
        norm = partial(normalize_specify_range, norm01=norm01)
    raw_transform = get_raw_transform(  # pyright: ignore[reportUnknownVariableType]
        normalizer=norm
    )
    return raw_transform  # pyright: ignore


def get_unsupervised_dataset(
    paths: List[str],
    raw_key: str,
    patch_shape: Tuple[int, ...],
    roi: Optional[Union[slice, Tuple[slice, ...]]] = None,
    n_samples: Optional[int] = None,
    global_stats: Optional[Dict[str, Dict[str, Any]]] = None,
    norm01: Optional[bool] = False,
) -> ConcatDataset[RawDataset]:
    transform = get_augmentations(ndim=len(patch_shape))  # Flips

    augmentations = (  # pyright: ignore[reportUnknownVariableType]
        weak_augmentations(),
        weak_augmentations(),
    )
    datasets: List[RawDataset] = []
    for path in paths:
        raw_transform = _build_raw_transform(path, global_stats, norm01)
        datasets.append(
            RawDataset(
                path,
                raw_key,
                patch_shape,
                raw_transform,
                transform,
                augmentations=augmentations,
                roi=roi,
                n_samples=n_samples,
            )
        )
    ds: ConcatDataset[RawDataset] = ConcatDataset(datasets)
    return ds


def get_DummySelfTraining_dataset(
    paths: List[str],
    raw_key: str,
    label_key: str,
    patch_shape: Tuple[int, ...],
    roi: Optional[Union[slice, Tuple[slice, ...]]] = None,
    n_samples: Optional[int] = None,
    global_stats: Optional[Dict[str, Dict[str, Any]]] = None,
    norm01: Optional[bool] = False,
) -> ConcatDataset[DummySelfTrainingDataset]:
    # raw_transform = None
    transform = get_augmentations(ndim=3)  # Flips

    augmentations = (  # pyright: ignore[reportUnknownVariableType]
        weak_augmentations(),
        weak_augmentations(),
    )
    datasets: List[DummySelfTrainingDataset] = []
    for path in paths:
        raw_transform = _build_raw_transform(path, global_stats, norm01)
        datasets.append(
            DummySelfTrainingDataset(
                raw_path=path,
                raw_key=raw_key,
                label_path=path,
                label_key=label_key,
                patch_shape=patch_shape,
                raw_transform=raw_transform,
                transform=transform,
                augmentations=augmentations,  # pyright: ignore[reportUnknownArgumentType]
                roi=roi,
                n_samples=n_samples,
            )
        )
    ds: ConcatDataset[DummySelfTrainingDataset] = ConcatDataset(datasets)
    return ds


def get_DummySelfTraining_loader(
    paths: List[str],
    raw_key: str,
    label_key: str,
    patch_shape: Tuple[int, ...],
    batch_size: int,
    num_workers: int = 8,
    n_samples: Optional[int] = None,
    roi: Optional[Union[slice, Tuple[slice, ...]]] = None,
    shuffle: bool = True,
    global_stats: Optional[Dict[str, Dict[str, Any]]] = None,
    norm01: Optional[bool] = False,
) -> torch.utils.data.DataLoader[Any]:

    ds = get_DummySelfTraining_dataset(
        paths,
        raw_key,
        label_key,
        patch_shape,
        roi=roi,
        n_samples=n_samples,
        global_stats=global_stats,
        norm01=norm01,
    )

    loader = get_data_loader(  # pyright: ignore[reportUnknownVariableType]
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
    )
    return loader  # pyright: ignore[reportUnknownVariableType]


def get_unsupervised_loader(
    paths: List[str],
    raw_key: str,
    patch_shape: Tuple[int, ...],
    batch_size: int,
    num_workers: int = 8,
    n_samples: Optional[int] = None,
    roi: Optional[Union[slice, Tuple[slice, ...]]] = None,
    shuffle: bool = True,
    global_stats: Optional[Dict[str, Dict[str, Any]]] = None,
    norm01: Optional[bool] = False,
) -> torch.utils.data.DataLoader[Any]:

    ds = get_unsupervised_dataset(
        paths,
        raw_key,
        patch_shape,
        roi=roi,
        n_samples=n_samples,
        global_stats=global_stats,
        norm01=norm01,
    )

    loader = get_data_loader(  # pyright: ignore[reportUnknownVariableType]
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
    )
    return loader  # pyright: ignore[reportUnknownVariableType]


def get_MT_unsupervised_loaders(
    cfg: SelfTrainingDataConfig,
    roi_unsupervised_train: Optional[Union[slice, Tuple[slice, ...]]] = None,
    roi_unsupervised_val: Optional[Union[slice, Tuple[slice, ...]]] = None,
    psuedo_labeler_name: psuedo_labeler_names = "default_pseudo_labeler",
):
    unsup_t_paths: List[str] = (  # pyright: ignore[reportUnknownVariableType]
        traverse_h5_paths(cfg.unsupervised_train_paths)
    )
    unsup_v_paths: List[str] = (  # pyright: ignore[reportUnknownVariableType]
        traverse_h5_paths(cfg.unsupervised_val_paths)
    )

    if cfg.global_normalisation:
        if cfg.global_percentiles is not None:
            min_p = cfg.global_percentiles[0]
            max_p = cfg.global_percentiles[1]
        else:
            min_p, max_p = None, None

        train_stats: Optional[Dict[str, Dict[str, Any]]] = {}
        val_stats: Optional[Dict[str, Dict[str, Any]]] = {}

        for train_path in unsup_t_paths:
            assert (
                Path(train_path).suffix == ".h5"
            ), "Global normalisation only designed for h5 files."
            raw = load_h5(train_path, cfg.raw_key)
            raw_stats = calculate_global_stats(
                raw, percentile_min=min_p, percentile_max=max_p
            )
            train_stats[train_path] = raw_stats

        for val_path in unsup_v_paths:
            assert (
                Path(val_path).suffix == ".h5"
            ), "Global normalisation only designed for h5 files."
            raw = load_h5(val_path, cfg.raw_key)
            raw_stats = calculate_global_stats(
                raw, percentile_min=min_p, percentile_max=max_p
            )
            val_stats[val_path] = raw_stats
    else:
        train_stats = None
        val_stats = None

    if psuedo_labeler_name == "direct_eval_pseudo_labeler":
        # For the dummy pseudo labeler, we use the DummySelfTrainingLoader
        assert (
            cfg.label_key is not None
        ), "label_key must be provided for DummySelfTrainingLoader"
        unsupervised_train_loader = get_DummySelfTraining_loader(
            unsup_t_paths,
            cfg.raw_key,
            cfg.label_key,
            cfg.patch_shape,
            cfg.batch_size,
            n_samples=cfg.n_samples_train,
            roi=roi_unsupervised_train,
            global_stats=train_stats,
            norm01=cfg.norm01,
        )
        unsupervised_val_loader = get_DummySelfTraining_loader(
            unsup_v_paths,
            cfg.raw_key,
            cfg.label_key,
            cfg.patch_shape,
            cfg.batch_size,
            n_samples=cfg.n_samples_val,
            roi=roi_unsupervised_val,
            global_stats=val_stats,
            norm01=cfg.norm01,
        )

    else:
        print("Get unsup loaders")
        unsupervised_train_loader = get_unsupervised_loader(
            unsup_t_paths,
            cfg.raw_key,
            cfg.patch_shape,
            cfg.batch_size,
            num_workers=cfg.num_workers,
            n_samples=cfg.n_samples_train,
            roi=roi_unsupervised_train,
            global_stats=train_stats,
            norm01=cfg.norm01,
        )
        unsupervised_val_loader = get_unsupervised_loader(
            unsup_v_paths,
            cfg.raw_key,
            cfg.patch_shape,
            cfg.batch_size,
            num_workers=cfg.num_workers,
            n_samples=cfg.n_samples_val,
            roi=roi_unsupervised_val,
            global_stats=val_stats,
            norm01=cfg.norm01,
        )

    return unsupervised_train_loader, unsupervised_val_loader
