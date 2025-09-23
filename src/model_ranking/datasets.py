from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Any,
    Tuple,
    Sequence,
    Literal,
    Union,
)
import warnings
from numpy.typing import NDArray
import os
import h5py  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
import glob
from itertools import chain
import torch
from torch.utils.data import Dataset, DataLoader
import skimage.morphology
from elf.wrapper import RoiWrapper  # pyright: ignore[reportMissingTypeStubs]


from pytorch3dunet.augment.transforms import StandardLabelToBoundary, Relabel
from pytorch3dunet.datasets.utils import (
    default_prediction_collate,  # pyright: ignore[reportUnknownVariableType]
)

from plantseg.functionals.dataprocessing import (  # pyright: ignore[reportMissingTypeStubs]
    set_background_to_value,  # pyright: ignore[reportUnknownVariableType]
)
from pytorch3dunet.unet3d.utils import (
    remove_background_seg,  # pyright: ignore[reportUnknownVariableType]
)

from model_ranking.utils import load_h5, get_roi_slice, is_ndarray, loader_classes
from model_ranking.dataclass import (
    EvalDatasetConfig,
    mito_dataset_type,
)
from torch_em.util.image import load_data
from torch_em.util.util import (
    ensure_tensor_with_channels,
    ensure_patch_shape,  # pyright: ignore[reportUnknownVariableType]
)
from torch_em.data.raw_dataset import (
    RawDataset,
)


def traverse_pred_files(file_paths: Sequence[str], save_postfix: str) -> List[str]:
    assert isinstance(file_paths, list)
    results: List[str] = []
    for file_path in file_paths:
        if os.path.isdir(file_path):
            # if file path is a directory take all H5 files in that directory
            iters = [glob.glob(os.path.join(file_path, f"*{save_postfix}.h5"))]
            for fp in chain(*iters):
                results.append(fp)
        else:
            results.append(file_path)
    return results


def get_datasets(
    config: mito_dataset_type,
    phase: Literal["train", "test"],
    output_path: Optional[str] = None,
    data_base_path: str = "/scratch/talks/data",
):

    if phase == "train":
        loader_cfg = config.train_loader.create_config(
            output_dir=output_path,
            data_base_path=data_base_path,
            phase=phase,
        )

    else:
        loader_cfg = config.loader.create_config(
            output_dir=output_path,
            data_base_path=data_base_path,
            phase=phase,
        )
    dataset_class = loader_classes(loader_cfg.dataset)
    datasets = dataset_class.create_datasets(loader_cfg.model_dump(), phase=phase)
    return datasets, loader_cfg


def get_loaders(
    config: mito_dataset_type,
    phase: Literal["train", "test"],
    output_path: Optional[str],
    shuffle: bool = True,
    data_base_path: str = "/scratch/talks/data",
):
    datasets, loader_cfg = get_datasets(
        config=config,
        phase=phase,
        output_path=output_path,
        data_base_path=data_base_path,
    )
    for dataset in datasets:
        if hasattr(dataset, "prediction_collate"):
            collate_fn = dataset.prediction_collate
        else:
            collate_fn = (  # pyright: ignore[reportUnknownVariableType]
                default_prediction_collate
            )
        yield DataLoader(
            dataset,
            batch_size=loader_cfg.batch_size,
            num_workers=loader_cfg.num_workers,
            shuffle=shuffle,
            collate_fn=collate_fn,
        )


class StandardEvalDataset(Dataset[Tuple[NDArray[Any], NDArray[Any]]]):
    def __init__(
        self,
        pred_path: str,
        gt_path: str,
        pred_key: str,
        gt_key: str,
        roi: Optional[List[List[int]]] = None,
        patch_key: str = "patch_index",
        ignore_index: Optional[int] = None,
        ignore_path: Optional[str] = None,
        ignore_key: Optional[str] = None,
        convert_to_binary_label: bool = False,
        convert_to_boundary_label: bool = False,
        relabel_background: bool = False,
        min_object_size: Optional[int] = None,
        instance_zero_background: bool = False,
        zero_largest_instance: bool = False,
    ):
        super().__init__()
        self.pred_path = pred_path
        self.gt_path = gt_path
        self.pred_key = pred_key
        self.gt_key = gt_key
        self.convert_to_binary_label = convert_to_binary_label
        self.convert_to_boundary_label = convert_to_boundary_label
        self.relabel_background = relabel_background
        self.instance_zero_background = instance_zero_background
        self.zero_largest_instance = zero_largest_instance
        self.min_obj_size = min_object_size
        if roi is not None:
            self.roi = get_roi_slice(roi)
        else:
            self.roi = None
        self.pred_patches = load_h5(self.pred_path, patch_key)
        self.ignore_index = ignore_index
        self.ignore_path = ignore_path
        self.ignore_key = ignore_key

        with h5py.File(self.gt_path, "r") as f:
            assert (
                self.gt_key in f
            ), f"Dataset {self.gt_key} not found in {self.gt_path}"
            ds = f[self.gt_key]
            assert isinstance(ds, h5py.Dataset)
            if self.roi is not None:
                self._gt: NDArray[Any] = ds[self.roi]
            else:
                self._gt: NDArray[Any] = ds[:]
            assert is_ndarray(self._gt), f"Data is not a numpy array: {self._gt}"

        with h5py.File(self.pred_path, "r") as f:
            assert (
                self.pred_key in f
            ), f"Dataset {self.pred_key} not found in {self.pred_path}"
            ds = f[self.pred_key]
            assert isinstance(ds, h5py.Dataset)
            self._pred: NDArray[Any] = ds[:]
            assert is_ndarray(self._pred), f"Data is not a numpy array: {self._pred}"

        if self.ignore_path is not None:
            assert (
                self.ignore_index is None
            ), "ignore_index is not allowed when ignore_path is provided"
            assert (
                self.ignore_key is not None
            ), f"ignore_key is required for ignore_path {self.ignore_path} is provided"
            with h5py.File(self.ignore_path, "r") as f:
                assert (
                    self.ignore_key in f
                ), f"Dataset {self.ignore_key} not found in {self.ignore_path}"
                ds = f[self.ignore_key]
                assert isinstance(ds, h5py.Dataset)
                if self.roi is not None:
                    self._ignore: Optional[NDArray[Any]] = ds[self.roi]
                else:
                    self._ignore: Optional[NDArray[Any]] = ds[:]
                assert is_ndarray(
                    self._ignore
                ), f"Data is not a numpy array: {self._ignore}"
        else:
            self._ignore = None

    def get_gt_patch(self, idx: tuple[slice, ...]) -> NDArray[Any]:
        # gt_patch = self._gt[0][idx]
        gt_patch = self._gt[idx].copy()
        # gt_patch = np.array(self._gt[idx].copy(), dtype=np.dtype[Any])
        # add channel dimension if necessary
        if len(gt_patch.shape) == 3:
            gt_patch = np.expand_dims(gt_patch, axis=0)
        return gt_patch

    def get_pred_patch(self, idx: int) -> NDArray[Any]:
        return self._pred[idx].copy()

    def get_gt_from_patchwise(self, index: int) -> NDArray[Any]:
        return self._gt[index].copy()

    def __getitem__(self, index: int) -> Tuple[NDArray[Any], NDArray[Any]]:
        pred = self.get_pred_patch(index)
        patch_slice = get_roi_slice(self.pred_patches[index])
        if self._gt.shape == self._pred.shape:
            gt = self.get_gt_from_patchwise(index)
        else:
            gt = self.get_gt_patch(patch_slice)
        if self._ignore is not None:
            # zero out ignore_index
            mask = self._ignore[patch_slice] == 1
            pred[mask] = 0
            gt[mask] = 0
        elif self.ignore_index is not None:
            # zero out ignore_index
            mask = gt == self.ignore_index
            pred[mask] = 0
            gt[mask] = 0
        if self.min_obj_size is not None:
            gt = skimage.morphology.remove_small_objects(  # pyright: ignore[reportUnknownVariableType]
                gt, min_size=self.min_obj_size
            )
            assert is_ndarray(gt), f"Data is not a numpy array: {gt}"
            if self.pred_key == "segmentation":
                pred = skimage.morphology.remove_small_objects(  # pyright: ignore[reportUnknownVariableType]
                    pred, min_size=self.min_obj_size
                )
                assert is_ndarray(pred), f"Data is not a numpy array: {pred}"
        if self.convert_to_boundary_label == True:
            gt = StandardLabelToBoundary(  # pyright: ignore[reportUnknownVariableType]
                ignore_index=self.ignore_index
            )(gt[0].copy())
            assert is_ndarray(gt), f"Data is not a numpy array: {gt}"
        if self.convert_to_binary_label == True:
            gt = (gt > 0).astype("uint8")
        if self.relabel_background == True:
            gt = set_background_to_value(  # pyright: ignore[reportUnknownVariableType]
                gt, 0
            )
            gt = Relabel()(gt[0])  # pyright: ignore[reportUnknownVariableType]
            assert is_ndarray(gt), f"Data is not a numpy array: {gt}"
            gt = np.expand_dims(gt, axis=0)
        if self.instance_zero_background == True:
            pred = remove_background_seg(  # pyright: ignore[reportUnknownVariableType]
                pred
            )
            assert is_ndarray(pred), f"Data is not a numpy array: {pred}"
        if self.zero_largest_instance == True:
            pred = (  # pyright: ignore[reportUnknownVariableType]
                set_background_to_value(pred, 0)
            )
            pred = Relabel()(pred[0])  # pyright: ignore[reportUnknownVariableType]
            assert is_ndarray(pred), f"Data is not a numpy array: {pred}"
            pred = np.expand_dims(pred, axis=0)
            assert is_ndarray(pred), f"Data is not a numpy array: {pred}"

        return pred, gt

    def __len__(self):
        return len(self.pred_patches)

    @classmethod
    def create_datasets(
        cls, dataset_config: EvalDatasetConfig
    ) -> List["StandardEvalDataset"]:
        pred_paths = traverse_pred_files(
            dataset_config.pred_path, getattr(dataset_config, "aug_name", "predictions")
        )
        # if no prediction files are found, try to traverse the directory for predictions
        if len(pred_paths) == 0:
            pred_paths = traverse_pred_files(dataset_config.pred_path, "predictions")
        gt_paths = traverse_pred_files(dataset_config.gt_path, "")
        # check for and remove metric_summary file paths
        gt_paths = [p for p in gt_paths if "metric_summary" not in p]
        assert len(pred_paths) == len(
            gt_paths
        ), f"Number of prediction files {len(pred_paths)} does not match number of ground truth files {len(gt_paths)}"

        datasets: List["StandardEvalDataset"] = []
        for i, pred_path in enumerate(pred_paths):

            pred_name = os.path.splitext(os.path.basename(pred_path))[0]
            if dataset_config.aug_name in pred_name:
                pred_name = pred_name.replace(f"_{dataset_config.aug_name}", "")
            else:
                pred_name = pred_name.replace("_predictions", "")
            assert (
                pred_name in os.path.splitext(os.path.basename(gt_paths[i]))[0]
            ), f"GT and pred file names do not match: {gt_paths[i]} != {pred_path}"
            dataset = cls(
                pred_path=pred_path,
                gt_path=gt_paths[i],
                pred_key=getattr(dataset_config, "pred_key", "predictions"),
                gt_key=dataset_config.gt_key,
                roi=getattr(dataset_config, "roi", None),
                patch_key=getattr(dataset_config, "patch_key", "patch_index"),
                ignore_index=getattr(dataset_config, "ignore_index", None),
                convert_to_binary_label=getattr(
                    dataset_config, "convert_to_binary_label", False
                ),
                convert_to_boundary_label=getattr(
                    dataset_config, "convert_to_boundary_label", False
                ),
                relabel_background=getattr(dataset_config, "relabel_background", False),
                min_object_size=getattr(dataset_config, "min_object_size", None),
                instance_zero_background=getattr(
                    dataset_config, "instance_zero_background", False
                ),
            )
            datasets.append(dataset)
        return datasets


class DummySelfTrainingDataset(RawDataset):
    def __init__(
        self,
        raw_path: Union[List[str], str, os.PathLike[str]],
        raw_key: Optional[str],
        label_path: Union[List[str], str],
        label_key: Optional[str],
        patch_shape: Tuple[int, ...],
        raw_transform: Optional[Callable[..., Any]] = None,
        transform: Optional[Callable[..., Any]] = None,
        roi: Optional[Union[slice, Tuple[slice, ...]]] = None,
        dtype: torch.dtype = torch.float32,
        n_samples: Optional[int] = None,
        sampler: Optional[Callable[..., Any]] = None,
        ndim: Optional[int] = None,
        with_channels: bool = False,
        augmentations: Optional[Tuple[Callable[..., Any], Callable[..., Any]]] = None,
    ):
        super().__init__(
            raw_path=raw_path,
            raw_key=raw_key,
            patch_shape=patch_shape,
            raw_transform=raw_transform,
            transform=transform,
            roi=roi,
            dtype=dtype,
            n_samples=n_samples,
            sampler=sampler,
            ndim=ndim,
            with_channels=with_channels,
            augmentations=augmentations,
        )
        self.label_path = label_path
        self.label_key = label_key
        self.label = load_data(label_path, label_key)

        if self.roi is not None:
            self.label = (
                RoiWrapper(self.label, (slice(None),) + self.roi)
                if self._with_channels
                else RoiWrapper(self.label, self.roi)
            )

    def __len__(self):
        return self._len

    def _get_sample(self, index: int):  # pyright: ignore
        if (self.raw is None) or (self.label is None):  # pyright: ignore
            raise RuntimeError(
                "DummySelfTrainingDataset has not been properly deserialized."
            )

        bb = self._sample_bounding_box()  # pyright: ignore[reportUnknownVariableType]
        assert isinstance(bb, tuple)
        raw = (  # pyright: ignore
            self.raw[(slice(None),) + bb]  # pyright: ignore
            if self._with_channels
            else self.raw[bb]  # pyright: ignore
        )
        label = (  # pyright: ignore
            self.label[(slice(None),) + bb]  # pyright: ignore
            if self._with_channels
            else self.label[bb]  # pyright: ignore
        )

        if self.sampler is not None:
            sample_id = 0
            while not self.sampler(raw):
                bb = self._sample_bounding_box()  # pyright: ignore
                raw = (  # pyright: ignore
                    self.raw[(slice(None),) + bb]  # pyright: ignore
                    if self._with_channels
                    else self.raw[bb]  # pyright: ignore
                )
                label = (  # pyright: ignore
                    self.label[(slice(None),) + bb]  # pyright: ignore
                    if self._with_channels
                    else self.label[bb]  # pyright: ignore
                )
                sample_id += 1
                if sample_id > self.max_sampling_attempts:
                    raise RuntimeError(
                        f"Could not sample a valid batch in {self.max_sampling_attempts} attempts"
                    )

        if self.patch_shape is not None:  # pyright: ignore
            raw = ensure_patch_shape(  # pyright: ignore
                raw=raw,  # pyright: ignore
                labels=None,
                patch_shape=self.patch_shape,
                have_raw_channels=self._with_channels,
            )
            label = ensure_patch_shape(  # pyright: ignore
                raw=label,  # pyright: ignore
                labels=None,
                patch_shape=self.patch_shape,
                have_raw_channels=self._with_channels,
            )

        # squeeze the singleton spatial axis if we have a spatial shape that is larger by one than self._ndim
        if len(self.patch_shape) == self._ndim + 1:
            raw = raw.squeeze(1 if self._with_channels else 0)  # pyright: ignore
            label = label.squeeze(1 if self._with_channels else 0)  # pyright: ignore

        return raw, label  # pyright: ignore

    def __getitem__(self, index: int):  # pyright: ignore
        raw, label = self._get_sample(index)  # pyright: ignore

        if self.raw_transform is not None:
            raw = self.raw_transform(raw)  # pyright: ignore

        if self.transform is not None:
            raw, label = self.transform(raw, label)  # pyright: ignore
            if isinstance(raw, list):
                assert len(raw) == 1  # pyright: ignore
                raw = raw[0]  # pyright: ignore

            if isinstance(label, list):
                assert len(label) == 1  # pyright: ignore
                label = label[0]  # pyright: ignore

            if self.trafo_halo is not None:
                raw = self.crop(raw)  # pyright: ignore
                label = self.crop(label)  # pyright: ignore

        raw = ensure_tensor_with_channels(
            raw, ndim=self._ndim, dtype=self.dtype  # pyright: ignore
        )
        label = ensure_tensor_with_channels(
            label, ndim=self._ndim, dtype=self.dtype  # pyright: ignore
        )

        if self.augmentations is not None:
            aug1, aug2 = self.augmentations  # pyright: ignore
            raw1, raw2 = aug1(raw), aug2(raw)  # pyright: ignore
            # concatenate raw1 with label
            raw1_label = torch.cat((raw1, label), dim=0)  # pyright: ignore
            return raw1_label, raw2  # pyright: ignore

        return raw, label

    def __getstate__(self):
        state = super().__getstate__()
        del state["label"]
        return state

    def __setstate__(self, state: Dict[str, Any]):
        super().__setstate__(state)

        label_path, label_key = state["label_path"], state["label_key"]
        roi = state["roi"]
        try:
            label = load_data(label_path, label_key)
            if roi is not None:
                label = (
                    RoiWrapper(label, (slice(None),) + roi)
                    if state["_with_channels"]
                    else RoiWrapper(label, roi)
                )
            state["label"] = label
        except Exception:
            msg = f"DummySelfTrainingDataset could not be deserialized because of missing {label_path}, {label_key}.\n"
            msg += "The dataset is deserialized in order to allow loading trained models from a checkpoint.\n"
            msg += "But it cannot be used for further training and wil throw an error."
            warnings.warn(msg)
            state["label"] = None

        self.__dict__.update(state)


def calculate_global_stats(
    img: Optional[Union[NDArray[Any], List[NDArray[Any]]]],
    skip: bool = False,
    percentile_min: Optional[float] = None,
    percentile_max: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Calculates the minimum percentile, maximum percentile, mean, and standard deviation of the image.

    Args:
        img: The input image array.
        skip: if True, skip the calculation and return None for all values.

    Returns:
        tuple[float, float, float, float]: The minimum percentile, maximum percentile, mean, and std dev
    """
    # if img is list, flatten and combine items of list
    if isinstance(img, list):
        img = np.concatenate([np.ravel(arr) for arr in img])
    if not skip:
        assert img is not None, "Image data cannot be None"
        mean = np.mean(img)
        std = np.std(img)
        min_val = np.min(img)
        max_val = np.max(img)
        if percentile_min is not None:
            pmin = np.percentile(img, percentile_min)
        else:
            pmin = None
        if percentile_max is not None:
            pmax = np.percentile(img, percentile_max)
        else:
            pmax = None

    else:
        pmin, pmax, mean, std, min_val, max_val = None, None, None, None, None, None

    return {
        "pmin": pmin,
        "pmax": pmax,
        "mean": mean,
        "std": std,
        "percentile_min": percentile_min,
        "percentile_max": percentile_max,
        "min": min_val,
        "max": max_val,
    }
