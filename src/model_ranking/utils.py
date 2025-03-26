import os
from typing import Optional, List, Sequence, Any, TypeGuard, Union
from pathlib import Path
from numpy.typing import NDArray
import h5py  # pyright: ignore[reportMissingTypeStubs]
import numpy as np

from pytorch3dunet.datasets.utils import (
    _loader_classes,  # pyright: ignore[reportUnknownVariableType, reportPrivateUsage]
)


def loader_classes(class_name: str):
    return _loader_classes(class_name)


def load_h5(
    path: str,
    key: str,
    roi: Optional[List[List[int]]] = None,
    select_index: Optional[Sequence[int]] = None,
) -> NDArray[Any]:
    # Load data
    assert os.path.exists(path), f"File {path} does not exist"
    # check that both roi and select_index are not provided
    assert not (roi and select_index), "Both roi and select_index cannot be provided"
    with h5py.File(path, "r") as f:
        ds = f[key]
        assert isinstance(ds, h5py.Dataset)
        if roi:
            data = ds[get_roi_slice(roi)]  # pyright: ignore[reportUnknownVariableType]
        elif select_index:
            data = ds[select_index]  # pyright: ignore[reportUnknownVariableType]
        else:
            data = ds[...]  # pyright: ignore[reportUnknownVariableType]
        assert is_ndarray(data), f"Data is not a numpy array: {data}"
    return data


def save_h5(
    save_path: Union[str, Path],
    out_key: str,
    data: NDArray[Any],
    overwrite: bool = False,
) -> None:
    with h5py.File(save_path, "a") as f:
        if out_key in f:
            if overwrite:
                del f[out_key]
            else:
                raise ValueError(f"Key {out_key} already exists in {save_path}")
        # if data is a float save as float32 in h5 format
        if data.shape == ():
            _ = f.create_dataset(out_key, data=data)
        else:
            _ = f.create_dataset(out_key, data=data, chunks=(1, *data.shape[1:]))


def get_roi_slice(roi: Sequence[Sequence[int]]) -> tuple[slice, ...]:
    # Create a tuple of slice objects based on the input list
    slices = tuple(slice(start, stop) for start, stop in roi)
    return slices


def is_ndarray(v: Any) -> TypeGuard[NDArray[Any]]:
    return isinstance(v, np.ndarray)
