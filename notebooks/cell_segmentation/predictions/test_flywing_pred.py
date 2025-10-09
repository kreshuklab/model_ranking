from model_ranking import load_h5, get_roi_slice
import numpy as np

from pytorch3dunet.unet3d.predictor import pmaps_to_IN_seg

path = "/g/kreshuk/talks/consistency_results/Instance_segmentation/Cells/FlyWing_to_FlyWing_gap/consistency/P_full_max_obj_4/fw_model_Unet2/norm_50_950/none/predictions/per03_predictions.h5"
Fw_gt_path = "/scratch/talks/data/FlyWing/GT/test/per03.h5"
label_key = "volumes/labels/expanded_cells_with_ignore"
raw_key = "volumes/raw"


id = 650
seg = load_h5(path, "segmentation", select_index=[id]).squeeze()
pred = load_h5(path, "predictions", select_index=[id]).squeeze()
patch_index = load_h5(path, "patch_index", select_index=[id])[0]


IN_seg = pmaps_to_IN_seg(
    pred, min_size=50, max_obj_size=3303, zero_large_instances=True
)
print(np.unique(IN_seg))
