from pathlib import Path
from typing import Optional, List

from model_ranking.utils import (
    load_h5,
    save_h5,
    copy_h5_dataset,
)


def convert_to_binary_labels(
    data_path: Path,
    save_path: Path,
    labels_key: str,
    roi: Optional[List[List[int]]] = None,
):

    data_labels = load_h5(data_path, labels_key, roi=roi)

    # Convert labels to binary: 0 stays 0, all other values become 1
    binary_labels = (data_labels != 0).astype(data_labels.dtype)
    save_h5(save_path, labels_key, binary_labels)


if __name__ == "__main__":
    data_path = Path("data/VNC/resized_pixels/val.h5")
    save_path = Path("data/VNC/resized_pixels/val_converted.h5")
    labels_key = "labels"
    raw_key = "raw"

    convert_to_binary_labels(
        data_path=data_path,
        save_path=save_path,
        labels_key=labels_key,
        roi=None,  # No ROI specified
    )

    copy_h5_dataset(
        source_path=data_path,
        target_path=save_path,
        dataset_name=raw_key,
    )
