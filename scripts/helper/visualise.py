import numpy as np
from model_ranking.visualise import h5_napari

if __name__ == "__main__":

    h5_napari(
        [
            "/g/kreshuk/talks/model_ranking_results/Self-Finetuning/Cells/Ovules_to_FlyWing_gap/default_selftraining/predictions/OvtoFw_def_ov_model_NA1/epoch-3/predictions/per03_predictions.h5"
        ]
        * 2,
        keys=["predictions", "segmentation"],
        names=["pred", "seg"],
        are_labels=[False, True],
        rois=[np.s_[50:55, :, :]] * 2,  # pyright: ignore
    )
