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

    h5_napari(
        [
            "/g/kreshuk/talks/consistency_results/Instance_segmentation/Cells/FlyWing_to_Ovules_gap/consistency/P1/fw_model8/norm_5_95/none/predictions/N_294_final_crop_ds2_none.h5",
            # "/g/kreshuk/talks/consistency_results/Instance_segmentation/Cells/FlyWing_to_Ovules_gap/consistency/P1/fw_model8/norm_5_95/gauss_a005-01/predictions/N_294_final_crop_ds2_gauss_a005-01.h5",
            "/g/kreshuk/talks/consistency_results/Instance_segmentation/Cells/FlyWing_to_Ovules_gap/consistency/P1/fw_model8/norm_5_95/gauss_a01-02/predictions/N_294_final_crop_ds2_gauss_a01-02.h5",
        ]
        + ["/scratch/talks/data/Ovules/GT2x/test/N_294_final_crop_ds2_patchwise.h5"]
        * 2,
        keys=["segmentation"] * 2 + ["raw", "label"],
        names=[
            "seg_none",
            # "seg_gauss_a005-01",
            "seg_gauss_a01-02",
            "raw",
            "labels",
        ],
        are_labels=[True] * 2 + [False, True],
        rois=None,
    )
