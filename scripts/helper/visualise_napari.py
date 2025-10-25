import napari
import numpy as np
from elf.io import open_file  # pyright: ignore

from typing import Any, Sequence, Union, Optional


def h5_napari(
    paths: Union[str, Sequence[str]],
    keys: Union[str, Sequence[str]],
    rois: Optional[Any] = None,
    names: Optional[Union[Sequence[str], Sequence[None]]] = None,
    are_labels: Optional[Sequence[bool]] = None,
):
    if isinstance(paths, str):
        paths = [paths]
    if isinstance(keys, str):
        keys = [keys]
    if rois is None:
        rois = [np.s_[:]] * len(paths)
    if names is None:
        names = [None] * len(paths)
    if are_labels is None:
        are_labels = [False] * len(paths)
    assert len(paths) == len(keys) == len(rois) == len(names) == len(are_labels)

    viewer = napari.Viewer()

    for path, key, roi, name, is_label in zip(paths, keys, rois, names, are_labels):
        # with h5py.File(path, 'r') as f:
        #     data = f[key][:].squeeze().astype(np.float32)
        data = open_file(path, "r")[key]  # pyright: ignore
        data = data[roi].squeeze()  # pyright: ignore

        if is_label:
            viewer.add_labels(data, name=name)  # pyright: ignore
        else:
            viewer.add_image(data, name=name)  # pyright: ignore

    napari.run()


if __name__ == "__main__":

    # h5_napari(
    #     [
    #         "/g/kreshuk/talks/consistency_results/Instance_segmentation/Cells/Ovules_to_Ovules_gap/consistency/P_fullslice/ov_model_NA1/norm_50_950/none/predictions/N_294_final_crop_ds2_predictions.h5",
    #         # "/g/kreshuk/talks/consistency_results/Instance_segmentation/Cells/Ovules_to_Ovules_gap/consistency/P_fullslice/ov_model_NA1/norm_50_950/gauss_a001-005/predictions/N_294_final_crop_ds2_predictions.h5",
    #         # "/g/kreshuk/talks/consistency_results/Instance_segmentation/Cells/Ovules_to_Ovules_gap/consistency/P_fullslice/ov_model_NA1/norm_50_950/gauss_a005-01/predictions/N_294_final_crop_ds2_predictions.h5",
    #         "/g/kreshuk/talks/consistency_results/Instance_segmentation/Cells/Ovules_to_Ovules_gap/consistency/P_fullslice/ov_model_NA1/norm_50_950/gauss_a01-015/predictions/N_294_final_crop_ds2_predictions.h5",
    #         "/g/kreshuk/talks/consistency_results/Instance_segmentation/Cells/Ovules_to_Ovules_gap/consistency/P_fullslice/ov_model_NA1/norm_50_950/gauss_a015-02/predictions/N_294_final_crop_ds2_predictions.h5",
    #         "/g/kreshuk/talks/consistency_results/Instance_segmentation/Cells/Ovules_to_Ovules_gap/consistency/P_fullslice/ov_model_NA1/norm_50_950/gauss_a02-025/predictions/N_294_final_crop_ds2_predictions.h5",
    #     ]
    #     + ["/scratch/talks/data/Ovules/GT2x/test/N_294_final_crop_ds2.h5"] * 2,
    #     keys=["segmentation"] * 4 + ["raw", "label"],
    #     names=[
    #         "seg_none",
    #         # "seg_gauss_a001-005",
    #         # "seg_gauss_a005-01",
    #         "seg_gauss_a01-015",
    #         "seg_gauss_a015-02",
    #         "seg_gauss_a02-025",
    #         "raw",
    #         "label_with_ignore",
    #     ],
    #     are_labels=[True] * 4 + [False, True],
    #     rois=None,
    # )

    # h5_napari(
    #     [
    #         "/g/kreshuk/talks/consistency_results/Instance_segmentation/Cells/Ovules_to_FlyWing_gap/consistency/P_fullslice/ov_model_UNet1/norm_50_950/none/predictions/per03_predictions.h5",
    #         # "/g/kreshuk/talks/consistency_results/Instance_segmentation/Cells/Ovules_to_FlyWing_gap/consistency/P_fullslice/ov_model_UNet1/norm_50_950/gauss_a001-005/predictions/per03_predictions.h5",
    #         # "/g/kreshuk/talks/consistency_results/Instance_segmentation/Cells/Ovules_to_FlyWing_gap/consistency/P_fullslice/ov_model_UNet1/norm_50_950/gauss_a005-01/predictions/per03_predictions.h5",
    #         "/g/kreshuk/talks/consistency_results/Instance_segmentation/Cells/Ovules_to_FlyWing_gap/consistency/P_fullslice/ov_model_UNet1/norm_50_950/gauss_a01-015/predictions/per03_predictions.h5",
    #         "/g/kreshuk/talks/consistency_results/Instance_segmentation/Cells/Ovules_to_FlyWing_gap/consistency/P_fullslice/ov_model_UNet1/norm_50_950/gauss_a015-02/predictions/per03_predictions.h5",
    #         "/g/kreshuk/talks/consistency_results/Instance_segmentation/Cells/Ovules_to_FlyWing_gap/consistency/P_fullslice/ov_model_UNet1/norm_50_950/gauss_a02-025/predictions/per03_predictions.h5",
    #     ]
    #     + ["/scratch/talks/data/FlyWing/GT/test/per03.h5"] * 2,
    #     keys=["segmentation"] * 4
    #     + ["volumes/raw", "volumes/labels/expanded_cells_with_ignore"],
    #     names=[
    #         "seg_none",
    #         # "seg_gauss_a001-005",
    #         # "seg_gauss_a005-01",
    #         "seg_gauss_a01-015",
    #         "seg_gauss_a015-02",
    #         "seg_gauss_a02-025",
    #         "raw",
    #         "label_with_ignore",
    #     ],
    #     are_labels=[True] * 4 + [False, True],
    #     rois=None,
    # )

    h5_napari(
        ["/scratch/talks/data/VNC/resized_pixels/source_mitoEM_true.h5"] * 2,
        keys=["resized_labels", "resized_raw"],
        names=["labels", "raw"],
        are_labels=[True, False],
        rois=[np.s_[:, :, :]] * 2,
    )
