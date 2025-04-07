import os
import fnmatch
from typing import Optional, List, Sequence, Any, TypeGuard, Union
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from numpy.typing import NDArray
import h5py  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
import re

from pytorch3dunet.datasets.utils import (
    _loader_classes,  # pyright: ignore[reportUnknownVariableType, reportPrivateUsage]
)


def loader_classes(class_name: str):
    return _loader_classes(class_name)


def load_h5(
    path: Union[str, Path],
    key: Union[str, Path],
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


def check_for_no_aug_configs(
    source_dataset: str,
    target_dataset: str,
    configs: List[Path],
) -> List[Path]:
    configs_with_NA = configs.copy()
    # check if configs already contains no_aug_config for this transfer
    NA_config_exists = False
    for config in configs:
        pattern = f"*/{source_dataset}_to_{target_dataset}_gap/*/none/none.yml"
        if fnmatch.fnmatch(str(config), pattern):
            NA_config_exists = True
            break
    if not NA_config_exists:
        # find single path in configs containing pattern f"{source_dataset}_to_{target_dataset}
        single_aug_path = next(
            (
                config
                for config in configs
                if f"{source_dataset}_to_{target_dataset}" in str(config)
            ),
            None,
        )
        assert single_aug_path is not None, "No single_aug_path found"
        # remove aug section and replace with none from single_aug_path
        no_aug_path = single_aug_path.parent.parent / "none" / "none.yml"
        configs_with_NA.append(no_aug_path)
    return configs_with_NA


def extract_filename(
    pred_path: Union[Path, str],
    suffix_names: List[str] = [
        "brt",
        "ctr",
        "gamma",
        "gauss",
        "none",
        "predictions",
        "DO",
        "FN",
        "FD",
    ],
) -> Optional[str]:
    if isinstance(pred_path, str):
        pred_path = Path(pred_path)
    filename = pred_path.stem
    # Create a regex pattern dynamically based on the aug_titles
    aug_pattern = "|".join(
        map(re.escape, suffix_names)
    )  # Escape to handle special characters
    pattern = rf"^(.*)_(?:{aug_pattern})(?:_.+)?$"

    match = re.match(pattern, filename)
    if match:
        return match.group(1)


def avoid_int_overflow(
    data: NDArray[Union[np.uint16, np.uint32, np.uint64]], max_value: int
) -> NDArray[Union[np.uint16, np.uint32, np.uint64]]:
    # check for overflow error
    if np.iinfo(data.dtype).max < max_value:
        for dtype in [np.uint16, np.uint32, np.uint64]:
            if np.iinfo(dtype).max >= max_value:
                # incease dtype size by one
                data = data.astype(dtype)
                break
        assert np.iinfo(data.dtype).max >= max_value, "Overflow error"
    return data


def generate_distinct_colors(n: int):
    hsv: Any = (  # pyright: ignore[reportUnknownVariableType]
        plt.cm.hsv  # pyright: ignore[reportAttributeAccessIssue]
    )
    hues = np.linspace(0, 1, n, endpoint=False)
    np.random.shuffle(hues)
    colors: List[Any] = [hsv(hue) for hue in hues]
    return colors


def get_unique_colourmap(data: NDArray[Any]) -> ListedColormap:
    """Generate a colormap with distinct colors for each unique value in the input data

    Args:
        data (np.ndarray): Input data

    Returns:
        ListedColormap: colourmap
    """
    num_unique_values = len(np.unique(data))
    colors = generate_distinct_colors(num_unique_values)
    colors[0] = (0, 0, 0, 1)  # Set the background color to black
    # Create a custom colormap
    return ListedColormap(colors)
