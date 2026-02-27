# type: ignore
import os
import numpy as np
from scipy.linalg import sqrtm
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Callable, Sequence, Tuple, Union
from tqdm import tqdm

from monai.data.utils import (
    compute_importance_map,
    dense_patch_slices,
    get_valid_patch_size,
)
from monai.utils import (
    BlendMode,
    PytorchPadMode,
    convert_data_type,
    fall_back_tuple,
    look_up_option,
)

# from CCFV.utils.evaluate import (
#     evaluate_2d,
# )

from pytorch3dunet.datasets.utils import (
    get_val_loader,  # pyright: ignore[reportUnknownVariableType]
)
from pytorch3dunet.unet3d.model import (
    get_model,  # pyright: ignore[reportUnknownVariableType]
)
from pytorch3dunet.unet3d.utils import (
    load_checkpoint,  # pyright: ignore[reportUnknownVariableType]
)


def cal_w_distance(f1, f2):
    miu_f1 = np.mean(f1, axis=0)
    miu_f2 = np.mean(f2, axis=0)
    cov_f1 = np.cov(f1.T)
    cov_f2 = np.cov(f2.T)

    # Add a small regularization term to the diagonals of the covariance matrices
    cov_f1 += np.eye(cov_f1.shape[0]) * 1e-6
    cov_f2 += np.eye(cov_f2.shape[0]) * 1e-6

    sqrt_cov_product = sqrtm(cov_f1.dot(cov_f2))

    # Ensure the result is real
    if np.iscomplexobj(sqrt_cov_product):
        sqrt_cov_product = np.real(sqrt_cov_product)

    delta_miu = miu_f1 - miu_f2
    w_d = (
        np.sum(delta_miu**2)
        + cov_f1.trace()
        + cov_f2.trace()
        - 2 * sqrt_cov_product.trace()
    )

    return np.sqrt(abs(w_d))


def cal_variety(matrix):
    eps = 1e-3
    row_norms = np.sum(matrix * matrix, axis=1)
    pairwise_inner_products = np.dot(matrix, matrix.T)
    pairwise_distances_squared = (
        np.expand_dims(row_norms, axis=1)
        + np.expand_dims(row_norms, axis=0)
        - 2 * pairwise_inner_products
    )
    pairwise_distances = 1 / np.sqrt(pairwise_distances_squared + eps)
    return (
        np.sum(np.triu(pairwise_distances, k=1))
        / len(pairwise_distances)
        / (len(pairwise_distances) - 1)
        * 2
    )


def evaluate_2d(config, test_loader, model):
    model.eval()
    layers = config["layers"]
    class_feature_dict = {
        layer: {j: [] for j in range(config["num_classes"] + 1)} for layer in layers
    }
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    global_feat_dict = {layer: [] for layer in layers}
    with torch.no_grad():
        for idx, (img, label) in tqdm(enumerate(test_loader), desc="Batch"):
            # print("evaluating index:{}".format(idx))
            # 2d image remove empty depth dimension
            img, label = img.to(device), label.to(device)
            img = torch.squeeze(img, dim=-3)
            label = torch.squeeze(label, dim=-3)
            feature_dict = sampling_2d_images(
                config["layers"],
                sample_num=config["sample_num"],
                inputs=img,
                labels=label,
                predictor=model,
            )
            for layer in class_feature_dict.keys():
                global_feat = np.concatenate(
                    [feat for feat in feature_dict[layer].values()], axis=0
                )
                global_feat = np.mean(global_feat, axis=0)
                global_feat_dict[layer].append(global_feat)
                for lb in class_feature_dict[layer].keys():
                    class_feature_dict[layer][lb].append(feature_dict[layer][lb])

    ccfv = 0
    for layer in class_feature_dict.keys():
        w_distance = 0.0
        global_feature = np.array(global_feat_dict[layer])
        var_f = cal_variety(global_feature)

        for lb in class_feature_dict[layer].keys():
            if lb == 0:
                continue
            length = len(class_feature_dict[layer][lb])
            for i in range(length):
                for j in range(i + 1, length):
                    w_distance += cal_w_distance(
                        class_feature_dict[layer][lb][i],
                        class_feature_dict[layer][lb][j],
                    )
            w_distance += w_distance / (length * length / 2) / config["num_classes"]
        ccfv += np.log(var_f / w_distance)

    print("ccfv:", ccfv)
    ccfv_norm = ccfv / len(class_feature_dict.keys())
    print("ccfv normalized by layer number:", ccfv_norm)
    return ccfv_norm


def sampling_2d_images(
    layers: list,
    sample_num: dict,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    predictor: Callable[
        ..., Union[torch.Tensor, Sequence[torch.Tensor], Dict[Any, torch.Tensor]]
    ],
):
    """function to sample pointwise feature vectors from layers of model for 2d input images
     without patching the input image.

    Args:
        features (Dict[str, np.ndarray]): features extracted by a network and saved per layer in a dictionary
        layers (list): layers of netwrork features were extracted from
        sample_num (dict): number of points to sample for each layer
        labels (torch.Tensor): input labels for set of patches (B, C, H, W)

    Returns:
        _type_: _description_
    """
    feat_extractor = FeatureExtractor(predictor, layers)
    features, _, _ = feat_extractor(inputs)
    feat_extractor.remove_handler()
    sample_dict = {
        layer: {j: [] for j in range(len(torch.unique(labels)))} for layer in layers
    }
    res_dict = {
        layer: {j: [] for j in range(len(torch.unique(labels)))} for layer in layers
    }
    for layer in layers:
        for lb in torch.unique(labels):
            lb_idx = torch.nonzero(labels == lb, as_tuple=False).cpu().numpy()
            np.random.seed(0)
            rand_idx = np.random.choice(
                len(lb_idx), min(len(lb_idx), sample_num[layer]), replace=False
            )
            sample_dict[layer][int(lb)] = lb_idx[rand_idx]

        seg_prob_out = features[layer]
        # output_shape equals the spatial dimensions of labels input which has shape (B, C, D, H, W)
        out_shape = labels.shape[2:]
        if seg_prob_out.shape != out_shape:
            seg_prob_out = F.interpolate(seg_prob_out, size=out_shape, mode="bilinear")
        for prob_out in seg_prob_out:
            for lb in sample_dict[layer].keys():
                visited = np.array([])
                for i in range(len(sample_dict[layer][lb])):
                    sample_idx = sample_dict[layer][lb][i]
                    point_feat = prob_out[:, sample_idx[2], sample_idx[3]].cpu().numpy()
                    res_dict[layer][lb].append(point_feat)
                    visited = np.append(visited, i)
                sample_dict[layer][lb] = np.delete(
                    sample_dict[layer][lb], visited.astype(np.int64), axis=0
                )
    for decoder in res_dict.keys():
        for lb in res_dict[decoder].keys():
            res_dict[decoder][lb] = np.array(res_dict[decoder][lb])
    return res_dict


def _get_scan_interval(
    image_size: Sequence[int],
    roi_size: Sequence[int],
    num_spatial_dims: int,
    overlap: float,
) -> Tuple[int, ...]:
    """
    Compute scan interval according to the image size, roi size and overlap.
    Scan interval will be `int((1 - overlap) * roi_size)`, if interval is 0,
    use 1 instead to make sure sliding window works.

    """
    if len(image_size) != num_spatial_dims:
        raise ValueError("image coord different from spatial dims.")
    if len(roi_size) != num_spatial_dims:
        raise ValueError("roi coord different from spatial dims.")

    scan_interval = []
    for i in range(num_spatial_dims):
        if roi_size[i] == image_size[i]:
            scan_interval.append(int(roi_size[i]))
        else:
            interval = int(roi_size[i] * (1 - overlap))
            scan_interval.append(interval if interval > 0 else 1)
    return tuple(scan_interval)


class FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, layers: list):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features_input = {layer: torch.empty(0) for layer in layers}
        self._features_output = {layer: torch.empty(0) for layer in layers}
        self.handlers = {layer: None for layer in layers}

        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            self.handlers[layer_id] = layer.register_forward_hook(
                self.save_outputs_hook(layer_id)
            )

    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, input, output):
            self._features_input[layer_id] = input[0]
            self._features_output[layer_id] = output

        return fn

    def forward(self, x):
        pred = self.model(x)
        features_output_copy = self._features_output.copy()
        features_input_copy = self._features_input.copy()
        return features_output_copy, features_input_copy, pred

    def remove_handler(self):
        for handler in self.handlers.values():
            handler.remove()


def run_ccfv_evaluation(config: Dict[str, Any]):
    model = get_model(config["model"])
    model_path = config["model_path"]
    load_checkpoint(
        model_path, model, model_key=config.get("model_key", "model_state_dict")
    )
    dataloaders = get_val_loader(  # pyright: ignore[reportUnknownVariableType]
        config["eval_dataloader"]
    )

    for loader in dataloaders:  # pyright: ignore[reportUnknownVariableType]
        ccfv_score = evaluate_2d(
            config["ccfv_config"],
            test_loader=loader,
            model=model,
        )
        # check if save file exists already
        if os.path.exists(config["ccfv_config"]["save_path"]):
            if config["ccfv_config"]["overwrite"]:
                np.save(config["ccfv_config"]["save_path"], ccfv_score)
            else:
                print(
                    f"File {config['ccfv_config']['save_path']} already exists, skipping"
                )
        else:
            np.save(config["ccfv_config"]["save_path"], ccfv_score)


def ms_sliding_window_sampling(
    layers: list,
    sample_num: dict,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    roi_size: Union[Sequence[int], int],
    sw_batch_size: int,
    predictor: Callable[
        ..., Union[torch.Tensor, Sequence[torch.Tensor], Dict[Any, torch.Tensor]]
    ],
    overlap: float = 0.0,
    mode: Union[BlendMode, str] = BlendMode.CONSTANT,
    sigma_scale: Union[Sequence[float], float] = 0.125,
    padding_mode: Union[PytorchPadMode, str] = PytorchPadMode.CONSTANT,
    cval: float = 0.0,
    sw_device: Union[torch.device, str, None] = None,
    device: Union[torch.device, str, None] = None,
    progress: bool = False,
    roi_weight_map: Union[torch.Tensor, None] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, ...], Dict[Any, torch.Tensor]]:

    compute_dtype = inputs.dtype
    num_spatial_dims = len(inputs.shape) - 2
    if overlap < 0 or overlap >= 1:
        raise ValueError("overlap must be >= 0 and < 1.")

    # determine image spatial size and batch size
    # Note: all input images must have the same image size and batch size
    batch_size, _, *image_size_ = inputs.shape

    if device is None:
        device = inputs.device
    if sw_device is None:
        sw_device = inputs.device

    roi_size = fall_back_tuple(roi_size, image_size_)
    # in case that image size is smaller than roi size
    image_size = tuple(
        max(image_size_[i], roi_size[i]) for i in range(num_spatial_dims)
    )
    pad_size = []
    for k in range(len(inputs.shape) - 1, 1, -1):
        diff = max(roi_size[k - 2] - inputs.shape[k], 0)
        half = diff // 2
        pad_size.extend([half, diff - half])
    inputs = F.pad(
        inputs,
        pad=pad_size,
        mode=look_up_option(padding_mode, PytorchPadMode),
        value=cval,
    )

    scan_interval = _get_scan_interval(image_size, roi_size, num_spatial_dims, overlap)

    # Store all slices in list
    slices = dense_patch_slices(image_size, roi_size, scan_interval)
    num_win = len(slices)  # number of windows per image
    total_slices = num_win * batch_size  # total number of windows

    # Create window-level importance map
    valid_patch_size = get_valid_patch_size(image_size, roi_size)
    if valid_patch_size == roi_size and (roi_weight_map is not None):
        importance_map = roi_weight_map
    else:
        try:
            importance_map = compute_importance_map(
                valid_patch_size, mode=mode, sigma_scale=sigma_scale, device=device
            )
        except BaseException as e:
            raise RuntimeError(
                "Seems to be OOM. Please try smaller patch size or mode='constant' instead of mode='gaussian'."
            ) from e
    importance_map = convert_data_type(
        importance_map, torch.Tensor, device, compute_dtype
    )[0]
    # handle non-positive weights
    min_non_zero = max(importance_map[importance_map != 0].min().item(), 1e-3)
    importance_map = torch.clamp(importance_map.to(torch.float32), min=min_non_zero).to(
        compute_dtype
    )

    # for each patch
    # labels = labels.as_tensor()
    feat_extractor = FeatureExtractor(predictor, layers)
    sample_dict = {
        layer: {j: [] for j in range(len(torch.unique(labels)))} for layer in layers
    }
    res_dict = {
        layer: {j: [] for j in range(len(torch.unique(labels)))} for layer in layers
    }

    for layer in layers:
        for lb in torch.unique(labels):
            lb_idx = torch.nonzero(labels == lb, as_tuple=False).numpy()
            np.random.seed(0)
            rand_idx = np.random.choice(
                len(lb_idx), min(len(lb_idx), sample_num[layer]), replace=False
            )
            sample_dict[layer][int(lb)] = lb_idx[rand_idx]
    # import pdb; pdb.set_trace()
    for slice_g in (
        tqdm(range(0, total_slices, sw_batch_size))
        if progress
        else range(0, total_slices, sw_batch_size)
    ):  ## progress=Fasle
        slice_range = range(slice_g, min(slice_g + sw_batch_size, total_slices))
        unravel_slice = [
            [slice(int(idx / num_win), int(idx / num_win) + 1), slice(None)]
            + list(slices[idx % num_win])
            for idx in slice_range
        ]
        window_data = torch.cat(
            [
                convert_data_type(inputs[win_slice], torch.Tensor)[0]
                for win_slice in unravel_slice
            ]
        ).to(sw_device)

        # window_data = torch.squeeze(window_data, dim=-3)
        features = feat_extractor(window_data)
        feat_extractor.remove_handler()

        out_shape = roi_size
        for layer in layers:
            seg_prob_out = features[layer][0]
            seg_prob_out = torch.unsqueeze(seg_prob_out, dim=-3)
            if seg_prob_out.shape != out_shape:
                # if seg_prob_out.shape != out_shape[1:]:
                seg_prob_out = F.interpolate(
                    seg_prob_out, size=out_shape, mode="trilinear"
                )
                # seg_prob_out = F.interpolate(seg_prob_out, size=out_shape[1:], mode='bilinear')
            for idx, prob_out in zip(unravel_slice, seg_prob_out):
                for lb in sample_dict[layer].keys():
                    visited = np.array([])
                    for i in range(len(sample_dict[layer][lb])):
                        sample_idx = sample_dict[layer][lb][i]
                        if (
                            idx[2].start <= sample_idx[2] < idx[2].stop
                            and idx[3].start <= sample_idx[3] < idx[3].stop
                            and idx[4].start <= sample_idx[4] < idx[4].stop
                        ):
                            point_feat = (
                                prob_out[
                                    :,
                                    sample_idx[2] - idx[2].start,
                                    sample_idx[3] - idx[3].start,
                                    sample_idx[4] - idx[4].start,
                                ]
                                .cpu()
                                .numpy()
                            )
                            # point_feat = prob_out[:, sample_idx[3]-idx[3].start, sample_idx[4]-idx[4].start].cpu().numpy()
                            # point_feat = prob_out[:, sample_idx[3]-idx[3].start, sample_idx[4]-idx[4].start].detach().numpy()
                            res_dict[layer][lb].append(point_feat)
                            visited = np.append(visited, i)
                    sample_dict[layer][lb] = np.delete(
                        sample_dict[layer][lb], visited.astype(np.int64), axis=0
                    )
    for decoder in res_dict.keys():
        for lb in res_dict[decoder].keys():
            res_dict[decoder][lb] = np.array(res_dict[decoder][lb])

    return res_dict


def evaluate_3d_own_dataloader(configs, test_loader, model):
    model.eval()
    print("begin evaluation!")
    layers = configs["layers"]
    class_feature_dict = {
        layer: {j: [] for j in range(configs["num_classes"] + 1)} for layer in layers
    }
    global_feat_dict = {layer: [] for layer in layers}
    with torch.no_grad():
        for idx, (raw, label) in enumerate(test_loader):
            print("evaluating index:{}".format(idx))
            # try:
            #    val_data["data"] = val_data["data"].permute(
            #        0, 1, 4, 3, 2).cuda()
            # except RuntimeError as e:     # cuda out of memory
            #    print("cuda out of memory occurred! Try to transfer the data to cpu!")
            # """inference"""
            # data_seg = val_data["seg"].permute(0, 1, 4, 3, 2).cuda()
            # data_seg = torch.nn.functional.interpolate(
            #    val_data['seg'], size=val_data["data"].shape[2:], mode="nearest")
            raw.cuda()
            label.cuda()
            layers = configs["layers"]
            feature_dict = ms_sliding_window_sampling(
                layers,
                configs["sample_num"],
                raw,
                label,
                [configs["roi_z"], configs["roi_y"], configs["roi_x"]],
                configs["sw_batch_size"],
                model,
                overlap=configs["infer_overlap"],
                mode=configs["window_mode"],
            )
            for layer in class_feature_dict.keys():
                global_feat = np.concatenate(
                    [feat for feat in feature_dict[layer].values()], axis=0
                )
                global_feat = np.mean(global_feat, axis=0)
                global_feat_dict[layer].append(global_feat)
                for lb in class_feature_dict[layer].keys():
                    class_feature_dict[layer][lb].append(feature_dict[layer][lb])
    ccfv = 0
    for layer in class_feature_dict.keys():
        w_distance = 0.0
        global_feature = np.array(global_feat_dict[layer])
        var_f = cal_variety(global_feature)

        for lb in class_feature_dict[layer].keys():
            if lb == 0:
                continue
            length = len(class_feature_dict[layer][lb])
            for i in range(length):
                for j in range(i + 1, length):
                    w_distance += cal_w_distance(
                        class_feature_dict[layer][lb][i],
                        class_feature_dict[layer][lb][j],
                    )
            w_distance += w_distance / (length * length / 2) / configs["num_classes"]
        ccfv += np.log(var_f / w_distance)

    print("ccfv:", ccfv)
