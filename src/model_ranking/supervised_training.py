import os
import json
from pathlib import Path

import numpy as np
import torch
import torch_em  # pyright: ignore[reportMissingTypeStubs]
from tqdm import tqdm
from typing import List, Tuple, Union, Optional, Any

from torch_em.transform import (  # pyright: ignore[reportMissingTypeStubs]
    BoundaryTransform,
    label,
    Compose,
    PadIfNecessary,
    get_augmentations,  # pyright: ignore[reportUnknownVariableType]
)
from torch_em.data.sampler import (  # pyright: ignore[reportMissingTypeStubs]
    MinInstanceSampler,
)

from elf.io import (  # pyright: ignore[reportMissingTypeStubs]
    open_file,  # pyright: ignore[reportUnknownVariableType]
)

from model_ranking.utils import (
    is_ndarray,
)


def _compute_rois(
    paths: List[str], key: str, root: Union[str, Path], patch_shape: Tuple[int, ...]
) -> List[Tuple[List[int], List[int]]]:

    def _roi(path: str, key: str):
        with open_file(path, "r") as f:  # pyright: ignore[reportUnknownVariableType]
            assert key in f, f"Could not find {key} in {path}"
            labels = f[key][:]  # pyright: ignore[reportUnknownVariableType]
            assert is_ndarray(labels), f"Labels are not a ndarray: {labels}"
            coords = np.where(labels != 0)
            start = [int(coord.min()) for coord in coords]
            stop = [int(coord.max()) + 1 for coord in coords]

            crop_shape = tuple(sto - sta for sta, sto in zip(start, stop))
            if any(csh < psh for csh, psh in zip(crop_shape, patch_shape)):
                shape_extender = [
                    (psh - csh) // 2 + 1 if psh > csh else 0
                    for csh, psh in zip(crop_shape, patch_shape)
                ]
                start = [sta - ext for sta, ext in zip(start, shape_extender)]
                stop = [sto + ext for sto, ext in zip(stop, shape_extender)]
                if any(sta < 0 for sta in start):
                    start = [max(sta, 0) for sta in start]
                    stop = [
                        sto + ext if sta == 0 else sto
                        for sta, sto, ext in zip(start, stop, shape_extender)
                    ]

                crop_shape = tuple(sto - sta for sta, sto in zip(start, stop))
            assert all(csh >= psh for csh, psh in zip(crop_shape, patch_shape))

            bb = tuple(slice(sta, sto) for sta, sto in zip(start, stop))
            labels = labels[bb]

        return start, stop

    roi_folder = os.path.join(root, "rois")
    os.makedirs(roi_folder, exist_ok=True)
    rois = []
    for path in tqdm(paths, desc="Computing rois"):
        ds_name, fname = Path(path).parts[-2], Path(path).stem
        out_path = os.path.join(
            roi_folder, f"{ds_name}_{fname}_{key.replace('/', '-')}.json"
        )

        if os.path.exists(out_path):
            with open(out_path) as f:
                this_roi = json.load(f)["roi"]
        else:
            this_roi = _roi(path, key)
            with open(out_path, "w") as f:
                json.dump({"roi": this_roi}, f)

        rois.append(this_roi)

    # for roi in rois:
    #     shape = tuple(roi[1][i] - roi[0][i] for i in range(3))
    #     print(shape)
    return rois  # pyright: ignore[reportUnknownVariableType]


def get_supervised_loader(
    data_paths: List[str],
    raw_key: str,
    label_key: str,
    patch_shape: Tuple[int, ...],
    batch_size: int,
    root: str,
    num_workers: int = 8,
    n_samples: Optional[int] = None,
    crop_to_labels: bool = False,  # NOTE: war vorher True
    add_boundary_transform: bool = True,
    label_dtype: torch.dtype = torch.float32,
) -> torch.utils.data.DataLoader[Any]:

    if crop_to_labels:
        rois_from_labels = _compute_rois(data_paths, label_key, root, patch_shape)
        rois = [
            tuple(slice(sta, sto) for sta, sto in zip(start, stop))
            for start, stop in rois_from_labels
        ][0]

    else:
        rois = None

    if add_boundary_transform:
        label_transform = BoundaryTransform(add_binary_target=True)
    else:
        label_transform = (  # pyright: ignore[reportUnknownVariableType]
            label.connected_components
        )
    transform = Compose(
        PadIfNecessary(patch_shape),
        get_augmentations(3),
    )

    sampler = MinInstanceSampler(min_num_instances=4)
    loader = torch_em.default_segmentation_loader(  # pyright: ignore[reportUnknownVariableType]
        data_paths,
        raw_key,
        data_paths,
        label_key,
        rois=rois,
        sampler=sampler,
        batch_size=batch_size,
        patch_shape=patch_shape,
        is_seg_dataset=True,
        label_transform=label_transform,
        transform=transform,
        num_workers=num_workers,
        shuffle=True,
        n_samples=n_samples,
        label_dtype=label_dtype,  # pyright: ignore[reportArgumentType]
    )
    return loader  # pyright: ignore[reportUnknownVariableType]
