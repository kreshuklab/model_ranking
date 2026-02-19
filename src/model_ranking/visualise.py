import napari
import numpy as np
from typing import Union, List, Optional, Tuple

from elf.io import (  # pyright: ignore[reportMissingTypeStubs]
    open_file,  # pyright: ignore[reportUnknownVariableType]
)


def h5_napari(
    paths: Union[str, List[str]],
    keys: Union[str, List[str]],
    rois: Optional[Union[List[slice], Tuple[slice, ...]]] = None,
    names: Optional[Union[List[str], List[None]]] = None,
    are_labels: Optional[List[bool]] = None,
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
        data = open_file(path, "r")[key]  # pyright: ignore[reportUnknownVariableType]
        data = data[roi].squeeze()  # pyright: ignore[reportUnknownVariableType]

        if is_label:
            _ = viewer.add_labels(
                data, name=name  # pyright: ignore[reportUnknownArgumentType]
            )
        else:
            _ = viewer.add_image(
                data, name=name  # pyright: ignore[reportUnknownArgumentType]
            )

    napari.run()
