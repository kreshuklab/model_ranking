import os
import fnmatch
from typing import Optional, List, Sequence, Any, Tuple, TypeGuard, Union, Dict
from pathlib import Path
from matplotlib import colors
from natsort import natsorted
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from numpy.typing import NDArray
import h5py  # pyright: ignore[reportMissingTypeStubs]
from h5py import File  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
import re
import torch
from skimage.transform import resize  # pyright: ignore[reportUnknownVariableType]

from pytorch3dunet.datasets.utils import (
    get_class,  # pyright: ignore[reportUnknownVariableType]
)

MODEL_ABBREVIATIONS_TO_DATASET = {
    #### Mitochondria
    "E": "EPFL",
    "Hm": "Hmito",
    "Rm": "Rmito",
    "V": "VNC",
    "H": "Hmito",
    "R": "Rmito",
}


def loader_classes(class_name: str):
    modules = [
        "pytorch3dunet.datasets.hdf5",
        "pytorch3dunet.datasets.dsb",
        "pytorch3dunet.datasets.utils",
        "model_ranking.datasets",
    ]
    return get_class(class_name, modules)


def load_h5(
    path: Union[str, Path],
    key: Union[str, Path],
    roi: Optional[List[List[int]]] = None,
    select_index: Optional[Sequence[int]] = None,
) -> NDArray[Any]:
    # Load data
    assert os.path.exists(path), f"File {path} does not exist"
    # check that both roi and select_index are not provided
    # assert not (roi and select_index), "Both roi and select_index cannot be provided"
    with h5py.File(path, "r") as f:
        ds = f[key]
        assert isinstance(ds, h5py.Dataset)
        if roi is not None:
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
                print(
                    f"Warning: Key {out_key} already exists in {save_path}. Not overwriting."
                )
                return
        # if data is a float save as float32 in h5 format
        if data.shape == ():
            _ = f.create_dataset(out_key, data=data)
        else:
            _ = f.create_dataset(out_key, data=data, chunks=(1, *data.shape[1:]))


def copy_h5_dataset(
    source_path: Union[str, Path], target_path: Union[str, Path], dataset_name: str
):
    # Check if the target file exists
    if not os.path.exists(target_path):
        # Create a new target file
        with h5py.File(target_path, "w") as f:
            pass

    # Check if the dataset already exists in the target file
    with h5py.File(target_path, "r") as f:
        if dataset_name in f:
            raise ValueError(
                f"Dataset '{dataset_name}' already exists in the target file"
            )

    # Copy the dataset from the source file to the target file
    with h5py.File(source_path, "r") as source, h5py.File(target_path, "a") as target:
        source_dataset = source[dataset_name]
        _ = target.create_dataset(dataset_name, data=source_dataset)


def load_npz(path: Union[str, Path], key: str):
    data = np.load(path)
    if key not in data:
        raise KeyError(f"Key '{key}' not found in {path}")
    return data[key]


def get_roi_slice(roi: Sequence[Sequence[int]]) -> tuple[slice, ...]:
    # Create a tuple of slice objects based on the input list
    slices = tuple(slice(start, stop) for start, stop in roi)
    return slices


def is_ndarray(v: Any) -> TypeGuard[NDArray[Any]]:
    return isinstance(v, np.ndarray)


def is_torch_tensor(v: Any) -> TypeGuard[torch.Tensor]:
    return isinstance(v, torch.Tensor)


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
        data (NDArray[Any]): Input data

    Returns:
        ListedColormap: colourmap
    """
    num_unique_values = len(np.unique(data))
    colors = generate_distinct_colors(num_unique_values)
    colors[0] = (0, 0, 0, 1)  # Set the background color to black
    # Create a custom colormap
    return ListedColormap(colors)


def get_random_colors(labels: NDArray[Any]) -> colors.ListedColormap:
    """Generate a random color map for a label image.

    Args:
        labels: The labels.

    Returns:
        The color map.
    """
    unique_labels = np.unique(labels)
    have_zero = 0 in unique_labels
    cmap = [[0, 0, 0]] if have_zero else []
    cmap += np.random.rand(len(unique_labels), 3).tolist()
    cmap = colors.ListedColormap(cmap)
    return cmap


def find_transfer_from_pred_path(pred_path: str) -> str:
    match = re.search(r"/([^/]*_to_[^/]*)/", pred_path)
    if match:
        substring = match.group(1)
        if "_gap" in substring:
            # remove the "_gap" suffix
            substring = "_".join(substring.split("_")[:-1])
    else:
        substring = ""

    return substring


def threshold_patch_foreground_ratio(
    patches: NDArray[Any], gt: NDArray[Any], threshold: float
) -> Tuple[List[int], List[int]]:
    """Threshold patches based on foreground ratio in ground truth.
    saving patch id of all patches above and below threhold seperately

    Args:
        patches (NDArray[Any]): patch_locations
        gt (NDArray[Any]): GT Volume
        threshold (float): foreground ratio threshold

    Returns:
        Tuple[List[int], List[int]]: patch ids above and below threshold respectively
    """
    if np.min(gt) > 0:
        print("Ground truth background not 0. Relabelling to 0")
        gt = _relabel(gt)

    above_th_ids: List[int] = []
    below_th_ids: List[int] = []
    for i, patch in enumerate(patches):
        patch_slice = get_roi_slice(patch)
        foreground_ratio = np.sum(gt[patch_slice] > 0) / np.prod(gt[patch_slice].shape)
        if foreground_ratio > threshold:
            above_th_ids.append(i)
        else:
            below_th_ids.append(i)
    return above_th_ids, below_th_ids


def _relabel(input: NDArray[Any]) -> NDArray[Any]:
    _, unique_labels = np.unique(input, return_inverse=True)
    return unique_labels.reshape(input.shape)


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
    return None


def create_h5_dataset(
    file: File, key: str, data: NDArray[Any], overwrite: bool
) -> None:
    if key in file.keys():
        print(f"Key {key} already exists in file")
        if overwrite == True:
            print(f"Overwriting {key} in file")
            del file[key]
            _ = file.create_dataset(key, data=data)
    else:
        _ = file.create_dataset(key, data=data)


def load_select_prediction_scores(
    path: Path,
    score_save_key: str,
    select_patches: Optional[NDArray[Any]] = None,
) -> NDArray[Any]:
    scores = load_h5(path, score_save_key)
    if select_patches is not None:
        scores = scores[select_patches]
    return scores


def get_output_dir(
    source: str,
    target: str,
    model_name: str,
    output_folder: Optional[str] = "patchwise",
    approach: Optional[str] = "consistency",
    result_type: Optional[str] = "prediction",
    base_seg_folder: str = "/g/kreshuk/talks/domain_gap/experiments/patch_segmentation",
):
    assert Path(
        base_seg_folder
    ).exists(), f"Base segmentation folder {base_seg_folder} does not exist"

    if "segmentation_ModelSelection" in base_seg_folder:
        assert output_folder is not None, "output_folder cannot be None"
        assert approach is not None, "approach cannot be None"
        assert result_type is not None, "result_type cannot be None"
        output_path = str(
            Path(base_seg_folder)
            / f"{source}_to_{target}"
            / approach
            / model_name
            / output_folder
            / result_type
        )
    elif "AdaptiveBatchNorm" in base_seg_folder:
        assert result_type is not None, "result_type cannot be None"
        output_path = str(
            Path(base_seg_folder)
            / f"{source}_to_{target}_gap"
            / model_name
            / result_type
        )

    elif "SAM" in source:
        assert approach is not None, "approach cannot be None"
        if output_folder is not None:
            output_path = str(
                Path(base_seg_folder)
                / target
                / source
                / output_folder
                / approach
                / model_name
            )
        else:
            output_path = str(
                Path(base_seg_folder) / target / source / approach / model_name
            )

    else:
        assert approach is not None, "approach cannot be None"
        assert result_type is not None, "result_type cannot be None"
        if output_folder is not None:
            output_path = str(
                Path(base_seg_folder)
                / f"{source}_to_{target}_gap"
                / approach
                / result_type
                / model_name
                / output_folder
            )
        else:
            output_path = str(
                Path(base_seg_folder)
                / f"{source}_to_{target}_gap"
                / approach
                / result_type
                / model_name
            )

    # Create save folder if it doesn't exist
    # Path(output_path).mkdir(parents=True, exist_ok=True)
    return output_path


def get_output_paths(
    source: str,
    target: str,
    selected_augmentations: Dict[str, List[str]],
    selected_norms: Union[List[Tuple[float, float]], List[None]],
    source_model: str,
    output: str = "metric_summary.h5",
    approach: str = "feature_perturbation_consistency",
    result_folder: str = "exp1",
    base_dir_path: str = "/g/kreshuk/talks/domain_gap/experiments/patch_segmentation/",
) -> List[str]:
    paths: List[str] = []
    for aug, alphas in selected_augmentations.items():
        for norm in selected_norms:
            if norm == None:
                norm_foldername = "norm_Normalize"
            else:
                norm_foldername = f"norm_{str(norm[0])}_{str(norm[1]).replace('.', '')}"
            path = list(
                Path(base_dir_path).glob(
                    (
                        f"{source}_to_{target}_gap/{approach}/{result_folder}/"
                        f"{source_model}/{norm_foldername}"
                    )
                )
            )
            assert len(path) == 1, f"num paths found == {len(path)}"
            if aug == "none":
                out_path = list((path[0] / f"{aug}").glob(output))
                assert len(out_path) == 1, f"num paths found == {len(out_path)}"
                paths.append(str(out_path[0]))
            else:
                for alpha in alphas:
                    out_path = list((path[0] / f"{aug}_{alpha}").glob(output))
                    assert len(out_path) == 1, f"num paths found == {len(out_path)}"
                    paths.append(str(out_path[0]))
    return paths


def add_device_to_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add device to the config dictionary.
    """
    device = config.get("device", None)
    if device == "cpu":
        config["device"] = "cpu"
        return config
    if torch.cuda.is_available():
        config["device"] = "cuda"
    else:
        config["device"] = "cpu"
    return config


def find_finetuning_result_paths(
    model_names: List[str],
    approach: str = "feature_perturbation",
    output_folder: str = "predictions",
    base_path: Path = Path(
        "/g/kreshuk/talks/model_ranking_results/Self-Finetuning/Mitochondria"
    ),
) -> List[Path]:
    paths: List[Path] = []
    for model in model_names:
        transfer = model.split("_")[0]
        source = transfer.split("to")[0]
        target = transfer.split("to")[1]
        path = list(
            base_path.glob(
                f"{source}*_to_{target}*_gap/{approach}/{output_folder}/{model}"
            )
        )
        assert len(path) == 1, f"Found {len(path)} paths for {model} in {base_path}"
        paths.append(path[0])
    return paths


def identify_transfer(path: Union[Path, str]) -> Optional[str]:
    if isinstance(path, Path):
        path = str(path)
    match = re.search(r"[^/]*_to_[^/]*_gap", path)
    return match.group(0) if match else None


def find_selftraining_pred_paths(
    model_names: List[str],
    approach: str = "feature_perturbation",
    base_path: Path = Path(
        "/g/kreshuk/talks/model_ranking_results/Self-Finetuning/Mitochondria"
    ),
) -> List[Path]:
    paths: List[Path] = []
    for model in model_names:
        transfer = model.split("_")[0]
        source = transfer.split("to")[0]
        target = transfer.split("to")[1]
        path = list(
            base_path.glob(f"{source}*_to_{target}*_gap/{approach}/predictions/{model}")
        )
        assert len(path) == 1, f"Found {len(path)} paths for {model} in {base_path}"
        paths.append(path[0])
    return paths


def xy_resize_scaling(source: str, target: str) -> float:
    dataset_xy_pixel_size_nm: Dict[str, Union[int, float]] = {
        "epfl": 5,
        "mitoEM": 8,
        "VNC": 4.6,
    }
    xy_scaling = dataset_xy_pixel_size_nm[target] / dataset_xy_pixel_size_nm[source]
    return xy_scaling


def resize_data_label_pair(
    data: NDArray[Any], label: NDArray[Any], xy_scale: float, raw_order: int = 3
):
    resized_shape = (
        data.shape[0],
        int(np.round(data.shape[1] * xy_scale)),
        int(np.round(data.shape[2] * xy_scale)),
    )
    resized_volume = np.zeros(resized_shape, dtype=np.float32)
    resized_label = np.zeros(resized_shape, dtype=np.uint8)
    for z, (z_slice, z_slice_label) in enumerate(
        tqdm(zip(data, label), desc=f"resize per z-slice")
    ):
        resized_volume[z] = resize(
            z_slice, (resized_shape[1], resized_shape[2]), order=raw_order
        )
        resized_label[z] = resize(
            z_slice_label, (resized_shape[1], resized_shape[2]), order=0
        )
    return resized_volume, resized_label


def get_source_from_model_name(model_name: str) -> str:
    """
    Extract the source dataset from the model name.
    Assumes the model name is in the format 'source_abbrev_(model_postfix)'.
    """
    model_identifier = model_name.split("_")[0]
    if "to" in model_identifier:
        model_identifier = model_identifier[0]
    return MODEL_ABBREVIATIONS_TO_DATASET[model_identifier]


def find_batchnorm_pred_path(
    model_name: str,
    base_path: Union[Path, str],
    perturbation: str = "none",
    return_summary: bool = False,
    summary_postfix: str = "full",
) -> Path:
    transfer = model_name.split("_")[0]
    source, target = transfer.split("to")
    pred_dir_path = (
        Path(base_path)
        / (
            f"{MODEL_ABBREVIATIONS_TO_DATASET[source]}_to_"
            + f"{MODEL_ABBREVIATIONS_TO_DATASET[target]}_gap"
        )
        / model_name
        / "predictions"
        / perturbation
        / "predictions"
    )
    if return_summary:
        pred_path = list(pred_dir_path.glob(f"*metric_summary_{summary_postfix}.h5"))
    else:
        pred_path = list(pred_dir_path.glob("*predictions.h5"))
    assert (
        len(pred_path) == 1
    ), f"Found {len(pred_path)} prediction files for {model_name} at {pred_dir_path}"
    pred_path = pred_path[0]
    return pred_path


def load_predictions_transformers(
    model_name: str,
    TTA_aug: str,
    base_dir_path: Union[str, Path],
    prediction_key: str = "prediction",
    id_range: Optional[Tuple[int, int]] = None,
    file_identifier: str = "sample",
):
    pred_paths = natsorted(
        Path(base_dir_path).rglob(f"{model_name}/{TTA_aug}/**/{file_identifier}*.h5")
    )
    if id_range is not None:
        pred_paths = pred_paths[id_range[0] : id_range[1]]

    preds: List[NDArray[Any]] = []
    for path in pred_paths:
        with h5py.File(path, "r") as f:
            ds = f[prediction_key]
            assert isinstance(ds, h5py.Dataset)
            preds.append(ds[...])  # pyright: ignore[reportUnknownArgumentType]
    pred_cmb = np.stack(preds, axis=0)
    return pred_cmb, pred_paths


def aug_name_to_sigma_tuple(aug_name: str) -> tuple[float, float]:
    """
    Convert augmentation name to sigma values tuple.

    Parameters:
    -----------
    aug_name : str
        Augmentation name in format "aXXX-XXX" or "aXXX-aXXX"
        (e.g., "a001-003" or "a001-a003")

    Returns:
    --------
    tuple[float, float]
        Tuple of (min_sigma, max_sigma) values

    Examples:
    ---------
    >>> aug_name_to_sigma_tuple("a001-003")
    (0.01, 0.03)
    >>> aug_name_to_sigma_tuple("a001-a003")
    (0.01, 0.03)
    >>> aug_name_to_sigma_tuple("a005-007")
    (0.05, 0.07)
    >>> aug_name_to_sigma_tuple("a01-02")
    (0.1, 0.2)
    """
    # Remove the 'a' prefix
    sigma_part = aug_name[1:]

    # Split by dash to get the two values
    min_val, max_val = sigma_part.split("-")

    # Remove 'a' prefix from max_val if present
    if max_val.startswith("a"):
        max_val = max_val[1:]

    # Add decimal point after first zero for each value
    def add_decimal(val_str: str) -> float:
        if len(val_str) >= 2 and val_str[0] == "0":
            # Insert decimal point after first zero
            decimal_str = val_str[0] + "." + val_str[1:]
            return float(decimal_str)
        else:
            # For values like "1", "2", etc., treat as is
            return float(val_str)

    min_sigma = add_decimal(min_val)
    max_sigma = add_decimal(max_val)

    return (min_sigma, max_sigma)


def find_dataset_object_sizes(data: NDArray[Any]):
    all_object_counts: List[int] = []

    for i in range(data.shape[0]):
        unique, counts = np.unique(  # pyright: ignore[reportUnknownVariableType]
            data[i], return_counts=True
        )
        assert is_ndarray(unique)
        assert is_ndarray(counts)
        if -1 in unique:
            mask = unique != -1
            unique = unique[mask]
            counts = counts[mask]
        if 0 in unique:
            mask = unique != 0
            unique = unique[mask]
            counts = counts[mask]
        all_object_counts.extend(counts)

    median_object_size = np.median(all_object_counts)
    max_object_size = np.max(all_object_counts)

    return median_object_size, max_object_size, all_object_counts
