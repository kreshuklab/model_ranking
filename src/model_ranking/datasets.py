from typing import List, Optional, Any, Tuple, Sequence
from numpy.typing import NDArray
import os
import h5py  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
import glob
from itertools import chain
from torch.utils.data import Dataset
import skimage

from pytorch3dunet.augment.transforms import StandardLabelToBoundary, Relabel

from plantseg.dataprocessing import (  # pyright: ignore[reportMissingTypeStubs]
    set_background_to_value,  # pyright: ignore[reportUnknownVariableType]
)
from pytorch3dunet.unet3d.utils import (
    remove_background_seg,  # pyright: ignore[reportUnknownVariableType]
)


from model_ranking.utils import load_h5, get_roi_slice, is_ndarray
from model_ranking.dataclass import EvalDatasetConfig


def traverse_pred_files(file_paths: Sequence[str], aug_name: str) -> List[str]:
    assert isinstance(file_paths, list)
    results: List[str] = []
    for file_path in file_paths:
        if os.path.isdir(file_path):
            # if file path is a directory take all H5 files in that directory
            iters = [glob.glob(os.path.join(file_path, f"*{aug_name}.h5"))]
            for fp in chain(*iters):
                results.append(fp)
        else:
            results.append(file_path)
    return results


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
            dataset_config.pred_path, dataset_config.aug_name
        )
        gt_paths = traverse_pred_files(dataset_config.gt_path, "")
        datasets: List["StandardEvalDataset"] = []
        for i, pred_path in enumerate(pred_paths):
            assert (
                os.path.splitext(os.path.basename(gt_paths[i]))[0]
                in os.path.splitext(os.path.basename(pred_path))[0]
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
