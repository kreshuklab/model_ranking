import torch
from torchvision import transforms  # pyright: ignore[reportMissingTypeStubs]
from typing import Callable, Optional, Tuple, Union, List, Any


from torch_em.data import RawDataset  # pyright: ignore[reportMissingTypeStubs]
from torch_em.segmentation import (  # pyright: ignore[reportMissingTypeStubs]
    get_data_loader,  # pyright: ignore[reportUnknownVariableType]
)
from torch_em.transform.raw import (  # pyright: ignore[reportMissingTypeStubs]
    standardize,  # pyright: ignore[reportUnknownVariableType]
    GaussianBlur,
    AdditiveGaussianNoise,
)
from torch_em.transform import (  # pyright: ignore[reportMissingTypeStubs]
    get_raw_transform,  # pyright: ignore[reportUnknownVariableType]
    get_augmentations,  # pyright: ignore[reportUnknownVariableType]
)


def weak_augmentations(p: float = 0.75):  # pyright: ignore[reportUnknownParameterType]
    norm = standardize  # pyright: ignore[reportUnknownVariableType]
    assert isinstance(norm, Callable)
    aug = transforms.Compose(
        [  # pyright: ignore[reportUnknownArgumentType]
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


def get_unsupervised_loader(
    paths: List[str],
    raw_key: str,
    patch_shape: Tuple[int, ...],
    batch_size: int,
    num_workers: int = 8,
    n_samples: Optional[int] = None,
    roi: Optional[Union[slice, Tuple[slice, ...]]] = None,
) -> torch.utils.data.DataLoader[Any]:
    roi = None

    # Standardization
    raw_transform = get_raw_transform()  # pyright: ignore[reportUnknownVariableType]
    transform = get_augmentations(ndim=3)  # Flips

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
    ds = torch.utils.data.ConcatDataset(  # pyright: ignore[reportUnknownVariableType]
        datasets
    )

    loader = get_data_loader(  # pyright: ignore[reportUnknownVariableType]
        ds, batch_size=batch_size, num_workers=num_workers, shuffle=True
    )
    return loader  # pyright: ignore[reportUnknownVariableType]
