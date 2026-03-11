from pathlib import Path
from typing import Optional

from model_ranking.utils import (
    load_h5,
    save_h5,
)


def slice_and_split_data(
    data_path: Path,
    save_path: Path,
    raw_key: str,
    labels_key: str,
    train_slice: Optional[slice],
    val_slice: Optional[slice],
    test_slice: Optional[slice],
):

    data_raw = load_h5(data_path, raw_key)
    data_labels = load_h5(data_path, labels_key)

    if train_slice is not None:
        # Save raw train split
        save_h5(
            save_path / "train.h5",
            out_key="raw",
            data=data_raw[train_slice],
        )
        # Save labels train split
        save_h5(
            save_path / "train.h5",
            out_key="labels",
            data=data_labels[train_slice],
        )

    if val_slice is not None:
        # Save raw val split
        save_h5(
            save_path / "val.h5",
            out_key="raw",
            data=data_raw[val_slice],
        )
        # Save labels val split
        save_h5(
            save_path / "val.h5",
            out_key="labels",
            data=data_labels[val_slice],
        )

    if test_slice is not None:
        # Save raw test split
        save_h5(
            save_path / "test.h5",
            out_key="raw",
            data=data_raw[test_slice],
        )
        # Save labels test split
        save_h5(
            save_path / "test.h5",
            out_key="labels",
            data=data_labels[test_slice],
        )


if __name__ == "__main__":
    data_path = Path("/data/mitoEM/R_train_data.hdf5")
    save_path = Path("data/Rmito")
    raw_key = "/volumes/raw"
    labels_key = "/volumes/labels/mito_ids"
    # train_slice: Optional[slice] = slice(0, 300)
    train_slice = slice(0, 300)
    val_slice: Optional[slice] = slice(300, 350)
    test_slice = None

    slice_and_split_data(
        data_path=data_path,
        save_path=save_path,
        raw_key=raw_key,
        labels_key=labels_key,
        train_slice=train_slice,
        val_slice=val_slice,
        test_slice=test_slice,
    )
