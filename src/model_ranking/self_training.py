import torch
from torchvision import transforms  # pyright: ignore[reportMissingTypeStubs]
from typing import Callable, Optional, Tuple, Union, List, Any
from torch.utils.data import ConcatDataset

from model_ranking.datasets import DummySelfTrainingDataset
from model_ranking.augmentations import normalize_specify_range

from torch_em.data import RawDataset
from torch_em.segmentation import (
    get_data_loader,  # pyright: ignore[reportUnknownVariableType]
)
from torch_em.transform.raw import (
    # normalize,
    GaussianBlur,
    AdditiveGaussianNoise,
)
from torch_em.transform import (
    get_raw_transform,  # pyright: ignore[reportUnknownVariableType]
    get_augmentations,  # pyright: ignore[reportUnknownVariableType]
)


def weak_augmentations(p: float = 0.75):  # pyright: ignore[reportUnknownParameterType]
    norm = normalize_specify_range
    assert isinstance(norm, Callable)
    aug = transforms.Compose(
        [
            norm,
            transforms.RandomApply([GaussianBlur(sigma=(0, 2.5))], p=p),
            transforms.RandomApply(
                [
                    AdditiveGaussianNoise(
                        scale=(0, 0.15),
                        clip_kwargs=False,  # pyright: ignore[reportArgumentType]
                    )
                ],
            ),
        ]
    )
    return get_raw_transform(
        normalizer=norm, augmentation1=aug
    )  # pyright: ignore[reportUnknownVariableType]


def get_unsupervised_dataset(
    paths: List[str],
    raw_key: str,
    patch_shape: Tuple[int, ...],
    roi: Optional[Union[slice, Tuple[slice, ...]]] = None,
    n_samples: Optional[int] = None,
) -> ConcatDataset[RawDataset]:
    raw_transform = get_raw_transform(  # pyright: ignore[reportUnknownVariableType]
        normalizer=normalize_specify_range
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
) -> ConcatDataset[DummySelfTrainingDataset]:
    raw_transform = get_raw_transform(  # pyright: ignore[reportUnknownVariableType]
        normalizer=normalize_specify_range
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
) -> torch.utils.data.DataLoader[Any]:
    roi = None

    ds = get_DummySelfTraining_dataset(
        paths,
        raw_key,
        label_key,
        patch_shape,
        roi=roi,
        n_samples=n_samples,
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
) -> torch.utils.data.DataLoader[Any]:
    roi = None

    ds = get_unsupervised_dataset(
        paths,
        raw_key,
        patch_shape,
        roi=roi,
        n_samples=n_samples,
    )

    loader = get_data_loader(  # pyright: ignore[reportUnknownVariableType]
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
    )
    return loader  # pyright: ignore[reportUnknownVariableType]
