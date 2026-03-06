import numpy as np
from model_ranking.visualise import h5_napari

if __name__ == "__main__":

    h5_napari(
        ["/g/kreshuk/talks/data/epfl/resized_pixels/test.h5"] * 2,
        keys=[
            "raw",
            "labels",
        ],
        names=["raw", "GT"],
        are_labels=[False, True],
        rois=[np.s_[50:55, :, :]] * 2,  # pyright: ignore
    )
