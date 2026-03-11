from elf.io import open_file  # pyright: ignore
import numpy as np
import h5py  # pyright: ignore[reportMissingTypeStubs]
import os
from typing import Dict, Any

from model_ranking.utils import xy_resize_scaling, resize_data_label_pair, is_ndarray

config: Dict[str, Any] = {
    "path": "data/VNC/data_labeled_mito.h5",
    "source": "Hmito",
    "raw_key": "raw",
    "label_key": "label",
    "roi": np.s_[:, :, :],
    "out_path": "data/pixel_resize",
    "out_raw_key": "resized_raw",
    "out_label_key": "resized_labels",
}


def main():
    data = open_file(  # pyright: ignore[reportUnknownVariableType]
        config["path"], mode="r"
    )[config["raw_key"]][config["roi"]]
    label = open_file(  # pyright: ignore[reportUnknownVariableType]
        config["path"], mode="r"
    )[config["label_key"]][config["roi"]]
    source = config["source"]
    target = config["path"].split("/")[-2]

    # x_scale, y_scale = xy_resize_scaling(source=source, target=target)
    xy_scale = xy_resize_scaling(source=source, target=target)
    print(f"x_scale: {xy_scale}, y_scale: {xy_scale}")

    assert is_ndarray(data) and is_ndarray(label), "Data and label must be numpy arrays"

    resized_volume, resized_label = resize_data_label_pair(data, label, xy_scale)

    out_file = os.path.join(config["out_path"], f"{target}", f"source_{source}_true.h5")
    if not os.path.exists(os.path.dirname(out_file)):
        os.makedirs(os.path.dirname(out_file))

    with h5py.File(out_file, "a") as f:
        dset = f.create_dataset(
            config["out_raw_key"], data=resized_volume, compression="gzip"
        )
        _ = f.create_dataset(
            config["out_label_key"], data=resized_label, compression="gzip"
        )
        dset.attrs["roi"] = str(config["roi"])
        dset.attrs["x_scale"] = xy_scale
        dset.attrs["y_scale"] = xy_scale


if __name__ == "__main__":
    main()
