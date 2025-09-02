import numpy as np
from numpy.typing import NDArray
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union
from torch.utils.data import DataLoader
from torchvision.transforms import Compose  # pyright: ignore[reportMissingTypeStubs]

from .augmentations import (
    classification_geometric_TTAs,
    classification_raw_TTAs,
    AUGMENTATION_ABBREVIATIONS,
)
from .datasets import ClassificationFilteredDataset
from .dataclass import ClassificationTTALoaderConfig
from .utils import get_patch_positions

from model_ranking.utils import load_h5, is_ndarray

from torch_em.segmentation import get_data_loader  # pyright: ignore
from torch_em.transform.augmentation import (
    get_augmentations,  # pyright: ignore
    KorniaAugmentationPipeline,
)
from torch_em.transform.raw import (
    normalize,  # pyright: ignore
    get_raw_augmentations,  # pyright: ignore
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


def classification_TTA_loader_from_dataset(
    config: ClassificationTTALoaderConfig,
    patch_positions: NDArray[Any],
    raw_transform: Optional[Union[Compose, Callable[[Any], NDArray[Any]]]],
    transform: Optional[KorniaAugmentationPipeline],
) -> DataLoader[ClassificationFilteredDataset]:
    dataset_cfg = config.dataset
    ds = ClassificationFilteredDataset(
        raw_path=dataset_cfg.raw_path,
        raw_key=dataset_cfg.raw_key,
        mask_path=dataset_cfg.mask_path,
        mask_key=dataset_cfg.mask_key,
        patch_shape=dataset_cfg.patch_shape,
        patch_starts=patch_positions,
        ndim=config.ndim,
        raw_transform=raw_transform,
        repeat_patches=dataset_cfg.repeat_patches,
        transform=transform,
        n_samples=config.n_samples,
        random_seed=dataset_cfg.patch_rnd_seed,
        mask_return=dataset_cfg.mask_return_mode,
        patch_return=dataset_cfg.patch_return_mode,
    )

    return get_data_loader(  # pyright: ignore[reportUnknownVariableType]
        ds,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
    )


def get_classification_TTA_loaders(
    config: ClassificationTTALoaderConfig,
) -> Tuple[
    Dict[str, DataLoader[ClassificationFilteredDataset]], Dict[str, NDArray[Any]]
]:
    patch_positions = get_patch_positions(config.patch_position)

    loaders: Dict[str, DataLoader[ClassificationFilteredDataset]] = {}
    alphas: Dict[str, NDArray[Any]] = {}

    if config.TTAugmentations is None:
        loaders["None"] = classification_TTA_loader_from_dataset(
            config,
            patch_positions,
            raw_transform=normalize,  # pyright: ignore[reportUnknownArgumentType]
            transform=None,
        )
    else:
        if config.TTAugmentations.transform_params:
            applied_transforms = []
            for transform_param in config.TTAugmentations.transform_params:
                TT_transform = classification_geometric_TTAs(
                    config.TTAugmentations, transform_param
                )
                count = applied_transforms.count(transform_param[0])
                applied_transforms.append(transform_param[0])
                loaders[f"{transform_param[0]}_{count}"] = (
                    classification_TTA_loader_from_dataset(
                        config,
                        patch_positions,
                        raw_transform=normalize,  # pyright: ignore[reportUnknownArgumentType]
                        transform=TT_transform,
                    )
                )
        if config.TTAugmentations.raw_transform_params:
            applied_raw_transforms = []
            for raw_transform_param in config.TTAugmentations.raw_transform_params:
                aug_type = AUGMENTATION_ABBREVIATIONS[raw_transform_param[0]]
                alpha_range = raw_transform_param[1]["alpha"]
                aug_key = (
                    f"{aug_type}_a{str(alpha_range[0]).replace('.', '')}-"
                    f"{str(alpha_range[1]).replace('.', '')}"
                )
                TTA_alphas, TT_raw_transform = classification_raw_TTAs(
                    config.TTAugmentations, raw_transform_param
                )

                count = applied_raw_transforms.count(raw_transform_param[0])
                applied_raw_transforms.append(raw_transform_param[0])
                loaders[aug_key] = classification_TTA_loader_from_dataset(
                    config,
                    patch_positions,
                    raw_transform=TT_raw_transform,
                    transform=None,
                )
                alphas[aug_key] = TTA_alphas

    return loaders, alphas
