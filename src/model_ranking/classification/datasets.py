from elf.io import open_file  # pyright: ignore
import numpy as np
from numpy.typing import NDArray
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose  # pyright: ignore[reportMissingTypeStubs]
from typing import Any, Callable, Literal, Optional, overload, Sequence, Tuple, Union

from model_ranking.utils import is_ndarray
from torch_em.transform import (
    labels_to_binary,  # pyright: ignore[reportUnknownVariableType]
)
from torch_em.transform.augmentation import (
    KorniaAugmentationPipeline,
)
from torch_em.util.util import (
    ensure_tensor_with_channels,
)

TORCH_DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
}


def target_to_tensor(target: float, dtype: torch.dtype):
    return torch.tensor([target], dtype=dtype)


class ClassificationFilteredDataset(Dataset[Any]):
    max_sampling_attempts = 500

    def __init__(
        self,
        raw_path: str,
        raw_key: str,
        mask_path: str,
        mask_key: str,
        patch_shape: Sequence[int],
        patch_starts: NDArray[Any],
        raw_transform: Optional[Union[Compose, Callable[[Any], NDArray[Any]]]] = None,
        transform: Optional[KorniaAugmentationPipeline] = None,
        dtype: torch.dtype = torch.float32,
        n_samples: Optional[int] = None,
        repeat_patches: bool = False,
        ndim: Optional[int] = 2,
        random_seed: Optional[int] = None,
        mask_return: bool = False,
        patch_return: bool = False,
    ):
        super().__init__()
        self.raw_path = raw_path
        self.raw_key = raw_key
        self.raw = open_file(raw_path, mode="r")[raw_key]
        self.mask_path = mask_path
        self.mask_key = mask_key
        self.mask = open_file(mask_path, mode="r")[mask_key]
        self.mask_return = mask_return
        self.patch_return = patch_return
        self.repeat_patches = repeat_patches

        # assert is_ndarray(self.raw), f"Data is not a numpy array: {self.raw.dtype}"
        # assert is_ndarray(self.mask), f"Data is not a numpy array: {self.mask.dtype}"

        shape_raw = self.raw.shape  # pyright: ignore[reportUnknownVariableType]
        shape_mask = self.mask.shape  # pyright: ignore[reportUnknownVariableType]
        assert (
            shape_raw == shape_mask
        ), f"raw_shape:{shape_raw}, mask_shape: {shape_mask}"

        self.shape = shape_raw
        self._ndim = (
            len(shape_raw)  # pyright: ignore[reportUnknownArgumentType]
            if ndim is None
            else ndim
        )
        assert self._ndim in (
            2,
            3,
            4,
        ), f"Invalid data dimensions: {self._ndim}. Only 2d, 3d or 4d data is supported"
        assert len(patch_shape) in (
            self._ndim,
            self._ndim + 1,
        ), f"{patch_shape}, {self._ndim}"
        self.patch_shape = patch_shape
        self.dtype = dtype
        self.random_seed = random_seed

        self.raw_transform = raw_transform
        self.transform = transform
        self.max_len = len(patch_starts)
        # self.TTA_alphas = TTA_alphas

        self._len = self.max_len if n_samples is None else n_samples
        self.sample_shape = patch_shape
        self.patch_start_positions = self.patch_start_sample(
            patch_starts, self._len, self.random_seed, self.repeat_patches
        )
        assert len(self.patch_start_positions) == self._len

    def __len__(self):
        return self._len

    @property
    def ndim(self):
        return self._ndim

    @staticmethod
    def patch_start_sample(
        patch_positions: NDArray[Any],
        num_samples: int,
        random_seed: Optional[int],
        repeat_patches: bool,
    ):
        if random_seed is None:
            assert len(patch_positions) >= num_samples, "Not enough patches to sample"
            return patch_positions[:num_samples]
        else:
            r = np.random.RandomState(random_seed)
            return patch_positions[
                r.choice(
                    patch_positions.shape[0],
                    num_samples,
                    replace=repeat_patches,
                )
            ]

    def _sample_bounding_box(self, bb_start: Sequence[int]):
        return tuple(
            slice(start, start + psh) for start, psh in zip(bb_start, self.sample_shape)
        )

    @overload
    def _get_sample(
        self, index: int, mask_return: Literal[True], patch_return: Literal[False]
    ) -> Tuple[NDArray[Any], Literal[0, 1], NDArray[Any]]: ...

    @overload
    def _get_sample(
        self, index: int, mask_return: Literal[False], patch_return: Literal[True]
    ) -> Tuple[NDArray[Any], Literal[0, 1], NDArray[Any]]: ...

    @overload
    def _get_sample(
        self, index: int, mask_return: Literal[False], patch_return: Literal[False]
    ) -> Tuple[NDArray[Any], Literal[0, 1]]: ...

    def _get_sample(self, index: int, mask_return: bool, patch_return: bool) -> Union[
        Tuple[NDArray[Any], Literal[0, 1], NDArray[Any]],
        Tuple[NDArray[Any], Literal[0, 1]],
    ]:
        if self.raw is None or self.mask is None:
            raise RuntimeError(
                "ClassificationDataset has not been properly deserialized."
            )
        bb_start = self.patch_start_positions[index]
        bb = self._sample_bounding_box(bb_start)

        raw, mask = (  # pyright: ignore[reportUnknownVariableType]
            self.raw[bb],
            self.mask[bb],
        )
        assert is_ndarray(raw), f"Data is not a numpy array: {raw.dtype}"
        assert is_ndarray(mask), f"Data is not a numpy array: {mask.dtype}"

        # ensure greyscale image has 1 channel
        if len(self.patch_shape) == self._ndim:
            raw = np.expand_dims(raw, 0)

        mask = labels_to_binary(mask)  # pyright: ignore[reportUnknownVariableType]

        assert is_ndarray(mask), f"Data is not a numpy array: {mask.dtype}"

        if len(np.unique(mask)) == 1 and np.unique(mask)[0] == 0:
            label: int = 0
        else:
            label: int = 1

        if mask_return is True:
            return raw, label, mask
        if patch_return is True:
            return raw, label, bb_start
        else:
            return raw, label

    def __getitem__(self, index: int):
        # if self.TTA_alphas is not None:
        #     alpha = self.TTA_alphas[index]
        mask: Optional[Union[NDArray[Any], torch.Tensor]] = None
        bb_start: Optional[NDArray[Any]] = None

        if self.mask_return is True:
            raw, label, mask = self._get_sample(
                index, mask_return=True, patch_return=False
            )
            assert is_ndarray(mask), f"Data is not a numpy array: {mask.dtype}"

        elif self.patch_return is True:
            raw, label, bb_start = self._get_sample(
                index, mask_return=False, patch_return=True
            )

        else:
            raw, label = self._get_sample(index, mask_return=False, patch_return=False)

        if self.raw_transform is not None:
            raw = self.raw_transform(raw)  # pyright: ignore[reportUnknownVariableType]

        assert is_ndarray(raw), f"Data is not a numpy array: {raw.dtype}"

        if self.transform is not None:
            if self.mask_return is True:
                assert (
                    mask is not None
                ), "mask should not be None when mask_return is True"
                raw, mask = self.transform(raw, mask)
            else:
                raw = self.transform(raw)[0]

        raw = ensure_tensor_with_channels(raw, ndim=self._ndim, dtype=self.dtype)

        label = target_to_tensor(label, dtype=self.dtype)

        if self.mask_return is True:
            assert mask is not None, "mask should not be None when mask_return is True"
            mask = ensure_tensor_with_channels(mask, ndim=self._ndim, dtype=self.dtype)
            return raw, label, mask
        elif self.patch_return is True:
            assert (
                bb_start is not None
            ), "bb_start should not be None when patch_return is True"
            return raw, label, bb_start
        else:
            return raw, label
