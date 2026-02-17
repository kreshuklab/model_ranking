from functools import partial
import torch
from typing import Optional, Tuple, Union, List, Any, Dict
from torch.utils.data import ConcatDataset

from model_ranking.datasets import DummySelfTrainingDataset
from model_ranking.augmentations import (
    normalize_specify_range,
    weak_augmentations,
)

from torch_em.data import RawDataset
from torch_em.segmentation import (
    get_data_loader,  # pyright: ignore[reportUnknownVariableType]
)
from torch_em.transform import (
    get_raw_transform,  # pyright: ignore[reportUnknownVariableType]
    get_augmentations,  # pyright: ignore[reportUnknownVariableType]
)


def get_unsupervised_dataset(
    paths: List[str],
    raw_key: str,
    patch_shape: Tuple[int, ...],
    roi: Optional[Union[slice, Tuple[slice, ...]]] = None,
    n_samples: Optional[int] = None,
    global_stats: Optional[Dict[str, Any]] = None,
    norm01: Optional[bool] = False,
) -> ConcatDataset[RawDataset]:
    if global_stats is not None:
        if global_stats.get("pmin") is not None:
            minval = global_stats["pmin"]
            maxval = global_stats["pmax"]
        else:
            minval = global_stats["min"]
            maxval = global_stats["max"]
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

    transform = get_augmentations(ndim=len(patch_shape))  # Flips

    augmentations = (  # pyright: ignore[reportUnknownVariableType]
        weak_augmentations(),
        weak_augmentations(),
    )
    datasets = [
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
        for path in paths
    ]
    ds: ConcatDataset[RawDataset] = ConcatDataset(datasets)
    return ds


def get_DummySelfTraining_dataset(
    paths: List[str],
    raw_key: str,
    label_key: str,
    patch_shape: Tuple[int, ...],
    roi: Optional[Union[slice, Tuple[slice, ...]]] = None,
    n_samples: Optional[int] = None,
    global_stats: Optional[Dict[str, Any]] = None,
    norm01: Optional[bool] = False,
) -> ConcatDataset[DummySelfTrainingDataset]:
    if global_stats is not None:
        if global_stats.get("pmin") is not None:
            minval = global_stats["pmin"]
            maxval = global_stats["pmax"]
        else:
            minval = global_stats["min"]
            maxval = global_stats["max"]
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
    # raw_transform = None
    transform = get_augmentations(ndim=3)  # Flips

    augmentations = (  # pyright: ignore[reportUnknownVariableType]
        weak_augmentations(),
        weak_augmentations(),
    )
    datasets = [
        DummySelfTrainingDataset(
            raw_path=path,
            raw_key=raw_key,
            label_path=path,
            label_key=label_key,
            patch_shape=patch_shape,
            raw_transform=raw_transform,  # pyright: ignore[reportUnknownArgumentType]
            transform=transform,
            augmentations=augmentations,  # pyright: ignore[reportUnknownArgumentType]
            roi=roi,
            n_samples=n_samples,
        )
        for path in paths
    ]
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
    global_stats: Optional[Dict[str, Any]] = None,
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
    global_stats: Optional[Dict[str, Any]] = None,
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
