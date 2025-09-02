import numpy as np
from numpy.typing import NDArray
from typing import Optional, Sequence, Any
from torch.utils.data import DataLoader

from .datasets import ClassificationFilteredDataset
from model_ranking.utils import load_h5, is_ndarray

from torch_em.segmentation import get_data_loader  # pyright: ignore
from torch_em.transform.raw import (
    normalize,  # pyright: ignore
    get_raw_augmentations,  # pyright: ignore
)
from torch_em.transform.augmentation import (
    get_augmentations,  # pyright: ignore
)


def classification_dataloader(
    path: str,
    patch_position_path: str,
    patch_position_key: str,
    raw_key: str,
    mask_key: str,
    ndim: int = 2,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 8,
    n_samples: Optional[int] = None,
    transforms_params: Optional[Sequence[Sequence[Any]]] = None,
    raw_transform_params: Optional[Sequence[Sequence[Any]]] = None,
    patch_shape: Sequence[int] = (1, 128, 128),
    random_seed: Optional[int] = None,
    mask_return: bool = False,
    patch_return: bool = False,
    patch_positions: Optional[NDArray[Any]] = None,
    repeat_patches: bool = False,
    roi: Optional[NDArray[Any]] = None,
) -> DataLoader[Any]:
    if patch_positions is None:
        patch_positions = load_h5(patch_position_path, patch_position_key)
        assert is_ndarray(
            patch_positions
        ), f"Data is not a numpy array: {patch_positions.dtype}"
        if n_samples is not None:
            if random_seed is not None:
                rng = np.random.default_rng(random_seed)
                patch_positions = rng.choice(
                    patch_positions, size=n_samples, replace=False
                )
            else:
                patch_positions = np.random.choice(
                    patch_positions, size=n_samples, replace=False
                )

    if roi is not None:
        roi = np.array(roi)
        select_ids = np.all(
            (patch_positions >= roi[:, 0]) & (patch_positions <= roi[:, 1]), axis=1
        )
        patch_positions = patch_positions[select_ids]

    if transforms_params is not None:
        transforms = get_augmentations(
            ndim=ndim, transforms=transforms_params, default_augs=False
        )
    else:
        transforms = None
    if raw_transform_params is not None:
        raw_transform = get_raw_augmentations(transform_inputs=raw_transform_params)
    else:
        raw_transform = normalize  # pyright: ignore[reportUnknownVariableType]

    ds = ClassificationFilteredDataset(
        raw_path=path,
        raw_key=raw_key,
        mask_path=path,
        mask_key=mask_key,
        patch_shape=patch_shape,
        patch_starts=patch_positions,
        ndim=ndim,
        raw_transform=raw_transform,  # pyright: ignore[reportUnknownArgumentType]
        repeat_patches=repeat_patches,
        transform=transforms,
        n_samples=n_samples,
        random_seed=random_seed,
        mask_return=mask_return,
        patch_return=patch_return,
    )

    return get_data_loader(  # pyright: ignore[reportUnknownVariableType]
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
