from numpy.typing import NDArray
from typing import Any, Callable, Dict, Optional, Union
from torch.utils.data import DataLoader
from torchvision.transforms import Compose  # pyright: ignore[reportMissingTypeStubs]

from .augmentations import (
    classification_geometric_TTAs,
    AUGMENTATION_ABBREVIATIONS,
)
from .datasets import ClassificationFilteredDataset
from .dataclass import ClassificationLoaderConfig
from .utils import get_patch_positions

from torch_em.segmentation import get_data_loader  # pyright: ignore
from torch_em.transform.augmentation import (
    get_augmentations,  # pyright: ignore
    KorniaAugmentationPipeline,
)
from torch_em.transform.raw import (
    normalize,  # pyright: ignore
    get_raw_augmentations,
    get_single_TTA_raw_augmentation,
)


def get_classification_dataloader(
    config: ClassificationLoaderConfig,
) -> DataLoader[Any]:

    patch_cfg = config.patch_position
    patch_positions = get_patch_positions(patch_cfg)

    transforms = None
    raw_transform = normalize  # pyright: ignore[reportUnknownVariableType]

    if config.aug_config:
        aug_cfg = config.aug_config
        if aug_cfg.transform_params is not None:
            transforms = get_augmentations(
                ndim=aug_cfg.ndim,
                transforms=aug_cfg.transform_params,
                default_augs=False,
            )
        else:
            transforms = None
        if aug_cfg.raw_transform_params is not None:
            raw_transform = get_raw_augmentations(
                transform_inputs=aug_cfg.raw_transform_params
            )
        else:
            raw_transform = normalize  # pyright: ignore[reportUnknownVariableType]

    return classification_loader(
        config,
        patch_positions,
        raw_transform=raw_transform,  # pyright: ignore[reportUnknownArgumentType]
        transform=transforms,
    )


def classification_loader(
    config: ClassificationLoaderConfig,
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
        random_seed=dataset_cfg.patch_rnd_seed,
        n_samples=config.n_samples,
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
    config: ClassificationLoaderConfig,
) -> Dict[str, DataLoader[ClassificationFilteredDataset]]:
    patch_positions = get_patch_positions(config.patch_position)

    loaders: Dict[str, DataLoader[ClassificationFilteredDataset]] = {}

    if config.aug_config is None:
        loaders["None"] = classification_loader(
            config,
            patch_positions,
            raw_transform=normalize,  # pyright: ignore[reportUnknownArgumentType]
            transform=None,
        )
    else:
        if config.aug_config.transform_params:
            applied_transforms = []
            for transform_param in config.aug_config.transform_params:
                TT_transform = classification_geometric_TTAs(
                    config.aug_config, transform_param
                )
                count = applied_transforms.count(transform_param["name"])
                applied_transforms.append(transform_param["name"])
                loaders[f"{transform_param['name']}_{count}"] = classification_loader(
                    config,
                    patch_positions,
                    raw_transform=normalize,  # pyright: ignore[reportUnknownArgumentType]
                    transform=TT_transform,
                )
        if config.aug_config.raw_transform_params:
            applied_raw_transforms = []
            for raw_transform_param in config.aug_config.raw_transform_params:
                aug_type = AUGMENTATION_ABBREVIATIONS[raw_transform_param["name"]]
                alpha_range = raw_transform_param["params"]["alpha"]
                aug_key = (
                    f"{aug_type}_a{str(alpha_range[0]).replace('.', '')}-"
                    f"{str(alpha_range[1]).replace('.', '')}"
                )
                TT_raw_transform = get_single_TTA_raw_augmentation(raw_transform_param)

                count = applied_raw_transforms.count(raw_transform_param["name"])
                applied_raw_transforms.append(raw_transform_param["name"])
                loaders[aug_key] = classification_loader(
                    config,
                    patch_positions,
                    raw_transform=TT_raw_transform,
                    transform=None,
                )

    return loaders
