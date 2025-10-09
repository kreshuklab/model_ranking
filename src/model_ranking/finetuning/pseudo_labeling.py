import numpy as np
from numpy.typing import NDArray
import torch
from typing import Optional, Any, Tuple, Union, Literal, List

from model_ranking.dataclass import (
    Pytorch3DUnetModelConfig,
    SemanticSegmentationConfig,
    InstanceSegmentationConfig,
    segmentation_type,
)
from model_ranking.metrics import (
    get_mask,
    HammingDistanceEval,
    AdaptedRandErrorEval,
    EffectiveInvarianceEval,
    CrossEntropyEval,
    DifferenceImageEval,
    EntropyEval,
    KLDivergenceEval,
    MultiClassF1Eval,
)
from model_ranking.utils import (
    is_torch_tensor,
    is_ndarray,
)

from pytorch3dunet.augment.transforms import (
    Transformer,
)
from pytorch3dunet.unet3d.model import (
    get_model,  # pyright: ignore[reportUnknownVariableType]
)
from pytorch3dunet.unet3d.predictor import (
    pmaps_to_IN_seg,  # pyright: ignore[reportUnknownVariableType]
)

consistency_metrics = Union[
    HammingDistanceEval,
    AdaptedRandErrorEval,
    EffectiveInvarianceEval,
    CrossEntropyEval,
    DifferenceImageEval,
    EntropyEval,
    KLDivergenceEval,
]


class AbstractConsistencyPatchwisePseudoLabeler:
    """Compute pseudo labels based on model predictions, typically from a teacher model.

    Args:
        consistency_threshold: Threshold for accepting patches, if the patch consistency
            is above the threshold the full prediction will be used, otherwise the patch will
            be masked out.
        consistency_metric: Metric used to compute the consistency of the patches.
        mask_threshold: threshold to consider masked pixels only for consistency
            calculation.
        seg_params: Segmentation parameters for processing prediction.

    """

    def __init__(
        self,
        consistency_metric: consistency_metrics,
        mask_threshold: Optional[float] = None,
        consistency_threshold: Optional[float] = None,
        seg_params: segmentation_type = SemanticSegmentationConfig(),
        activation: Optional[torch.nn.Module] = None,
    ):
        super().__init__()
        self.consistency_metric = consistency_metric
        self.mask_threshold = mask_threshold
        self.consistency_threshold = consistency_threshold
        self.seg_params = seg_params
        # TODO serialize the class names and kwargs for activation instead
        self.consistency_log: List[float] = []
        self.activation = activation

    def _compute_label_mask(
        self, pseudo_labels: NDArray[Any], perturbed_pseudo_labels: NDArray[Any]
    ) -> Tuple[torch.Tensor, NDArray[Any]]:

        if isinstance(self.consistency_metric, AdaptedRandErrorEval):
            consis_score, consis_mask = self.consistency_metric(
                perturbed_pseudo_labels, pseudo_labels
            )
        else:
            if self.mask_threshold is None:
                consis_mask = np.ones_like(pseudo_labels)
            else:
                consis_mask = get_mask(
                    pseudo_labels, perturbed_pseudo_labels, self.mask_threshold
                )
            consis_score = self.consistency_metric(
                perturbed_pseudo_labels, pseudo_labels, consis_mask
            )
        mask = np.zeros_like(pseudo_labels)
        if isinstance(self.consistency_metric, HammingDistanceEval):
            # Hamming distance is a measure of dissimilarity, so we want to keep
            # the patches with a low distance
            ids = np.argwhere(consis_score < self.consistency_threshold)
            consis_score_PP = consis_score

        elif isinstance(self.consistency_metric, AdaptedRandErrorEval):
            ids = np.argwhere(consis_score[:, 0] > self.consistency_threshold)
            consis_score_PP = consis_score[:, 0]

        else:
            consis_score_PP = np.array(
                np.nanmean(consis_score, axis=tuple(range(1, consis_score.ndim)))
            )
            ids = np.argwhere(consis_score_PP > self.consistency_threshold)

        mask[ids] = 1

        for score in consis_score_PP:
            self.consistency_log.append(score)

        return torch.from_numpy(mask), consis_score_PP

    def get_instance_labels(
        self, pseudo_labels: torch.Tensor, pseudo_labels_perturbed: torch.Tensor
    ):
        assert isinstance(self.seg_params, InstanceSegmentationConfig), (
            "Segmentation type is not instance segmentation. "
            "Please use the correct segmentation type."
        )
        ps_labels = torch.zeros_like(pseudo_labels)
        ps_labels_perturbed = torch.zeros_like(pseudo_labels_perturbed)
        for i in range(len(pseudo_labels)):
            ps_labels[i] = (
                torch.from_numpy(
                    pmaps_to_IN_seg(
                        pseudo_labels[i].detach().cpu().numpy().squeeze(),
                        min_size=self.seg_params.min_size,
                        beta=self.seg_params.beta,
                        zero_largest_instance=self.seg_params.zero_largest_instance,
                        zero_large_instances=self.seg_params.zero_large_instances,
                        large_instance_multiplier=self.seg_params.large_instance_multiplier,
                    )
                )
                .to(pseudo_labels.dtype)
                .to(pseudo_labels.device)
            )
            ps_labels_perturbed[i] = (
                torch.from_numpy(
                    pmaps_to_IN_seg(
                        pseudo_labels_perturbed[i].detach().cpu().numpy().squeeze(),
                        min_size=self.seg_params.min_size,
                        zero_largest_instance=self.seg_params.zero_largest_instance,
                        beta=self.seg_params.beta,
                        zero_large_instances=self.seg_params.zero_large_instances,
                        large_instance_multiplier=self.seg_params.large_instance_multiplier,
                    )
                )
                .to(ps_labels_perturbed.dtype)
                .to(ps_labels_perturbed.device)
            )
        return ps_labels, ps_labels_perturbed

    def step(self, metric: Optional[float], epoch: int):
        pass

    def save_consistency_log(self, path: str):
        """Save the consistency log to a file."""
        np.savez(
            path,
            np.array(self.consistency_log, dtype=np.float32),
        )


class InputConsistencyPatchwisePseudoLabeler(AbstractConsistencyPatchwisePseudoLabeler):
    """Compute pseudo labels based on model predictions, typically from a teacher model.
    Optionally apply patch filter depending on the model's predictions consistency under
    input space perturbations.

    Args:
        transformer: Transformer to apply to the input during consistency analysis.
        consistency_threshold: Threshold for accepting patches, if the patch consistency
            is above the threshold the full prediction will be used, otherwise the patch will
            be masked out.
        mask_threshold: threshold to consider masked pixels only for consistency
            calculation.
        consistency_metric: Metric used to compute the consistency of the patches.
        seg_params: Segmentation parameters for processing prediction.

    """

    def __init__(
        self,
        transformer: Transformer,
        consistency_metric: consistency_metrics,
        mask_threshold: Optional[float] = None,
        consistency_threshold: Optional[float] = None,
        seg_params: segmentation_type = SemanticSegmentationConfig(),
        activation: Optional[torch.nn.Module] = None,
    ):
        super().__init__(
            consistency_metric=consistency_metric,
            mask_threshold=mask_threshold,
            consistency_threshold=consistency_threshold,
            seg_params=seg_params,
            activation=activation,
        )
        self.transform = transformer.raw_transform()

    def __call__(
        self, teacher: torch.nn.Module, input_: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # input_ = input_.squeeze(2)
        pseudo_labels = teacher(input_)
        perturbed_input_ = (
            torch.from_numpy(
                self.transform(input_.detach().cpu().numpy().astype("float32"))
            )
            .to(input_.dtype)
            .to(input_.device)
        )
        perturbed_pseudo_labels = teacher(perturbed_input_)
        assert is_torch_tensor(pseudo_labels), "pseudo_labels is not a torch.Tensor."
        assert is_torch_tensor(
            perturbed_pseudo_labels
        ), "pseudo_labels_perturbed is not a torch.Tensor."

        if self.activation is not None:
            pseudo_labels = self.activation(pseudo_labels)
            perturbed_pseudo_labels = self.activation(perturbed_pseudo_labels)

        if self.seg_params.name == "instance":
            pseudo_labels, perturbed_pseudo_labels = self.get_instance_labels(
                pseudo_labels, perturbed_pseudo_labels
            )

        if self.consistency_threshold is None:
            label_mask = None
        else:
            ps_lab = pseudo_labels.detach().cpu().numpy().astype("float32")
            ps_lab_perturbed = (
                perturbed_pseudo_labels.detach().cpu().numpy().astype("float32")
            )
            assert is_ndarray(ps_lab)
            assert is_ndarray(ps_lab_perturbed)
            label_mask, _ = self._compute_label_mask(
                ps_lab,
                ps_lab_perturbed,
            )
            label_mask = label_mask.to(input_.device)
        assert is_torch_tensor(pseudo_labels), (
            "pseudo_labels is not a torch.Tensor. "
            "Either pseudo_labels or label_mask must be a torch.Tensor."
        )
        return pseudo_labels, label_mask


class ModelConsistencyPatchWisePseudoLabeler(AbstractConsistencyPatchwisePseudoLabeler):
    """Compute pseudo labels based on model predictions, typically from a teacher model.
    Optionally apply patch filter depending on the model's predictions consistency under
    feature space perturbations.

    Args:
        perturbed_model_config: Configuration for the perturbed model.
        consistency_metric: Metric used to compute the consistency of the patches.
        mask_threshold: threshold to consider masked pixels only for
            consistency calculation.
        consistency_threshold: Threshold for accepting patches, if the patch consistency
            is above the threshold the full prediction will be used, otherwise the patch
            will be masked out.
        seg_params: Segmentation parameters for processing prediction.
    """

    def __init__(
        self,
        perturbed_model_config: Pytorch3DUnetModelConfig,
        consistency_metric: consistency_metrics,
        mask_threshold: Optional[float] = None,
        consistency_threshold: Optional[float] = None,
        seg_params: segmentation_type = SemanticSegmentationConfig(),
        activation: Optional[torch.nn.Module] = None,
    ):
        super().__init__(
            consistency_metric=consistency_metric,
            mask_threshold=mask_threshold,
            consistency_threshold=consistency_threshold,
            seg_params=seg_params,
            activation=activation,
        )
        self.perturbed_teacher = get_model(perturbed_model_config.model_dump())
        # self.log_pseudo_labels: List[NDArray[Any]] = []
        # self.log_pseudo_labels_perturbed: List[NDArray[Any]] = []

    def __call__(
        self, teacher: torch.nn.Module, input_: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        pseudo_labels = teacher(input_)
        _ = self.perturbed_teacher.load_state_dict(teacher.state_dict())
        perturbed_teacher = self.perturbed_teacher.to(next(teacher.parameters()).device)
        perturbed_teacher = perturbed_teacher.eval()
        perturbed_pseudo_labels = perturbed_teacher(input_)

        if self.activation is not None:
            pseudo_labels = self.activation(pseudo_labels)
            perturbed_pseudo_labels = self.activation(perturbed_pseudo_labels)

        if self.seg_params.name == "instance":
            pseudo_labels, perturbed_pseudo_labels = self.get_instance_labels(
                pseudo_labels, perturbed_pseudo_labels
            )

        if self.consistency_threshold is None:
            label_mask = None
        else:
            ps_lab = pseudo_labels.detach().cpu().numpy().astype("float32")
            # self.log_pseudo_labels.append(ps_lab)
            ps_lab_perturbed = (
                perturbed_pseudo_labels.detach().cpu().numpy().astype("float32")
            )
            # self.log_pseudo_labels_perturbed.append(ps_lab_perturbed)
            assert is_ndarray(ps_lab)
            assert is_ndarray(ps_lab_perturbed)
            label_mask, _ = self._compute_label_mask(
                ps_lab,
                ps_lab_perturbed,
            )
            label_mask = label_mask.to(input_.device)
        assert is_torch_tensor(pseudo_labels), (
            "pseudo_labels is not a torch.Tensor. "
            "Either pseudo_labels or label_mask must be a torch.Tensor."
        )
        return pseudo_labels, label_mask


class ScheduledPseudoLabeler:
    """
    This class implements a scheduled pseudo-labeling mechanism, where pseudo labels
    are generated from a teacher model's predictions, and the confidence threshold
    for filtering the pseudo labels can be adjusted over time based on a performance
    metric or a fixed schedule. It includes options for adjusting thresholds from
    both sides (for binary classification) or from one side (for multiclass problems).
    The threshold can be dynamically reduced to improve the quality of the pseudo labels
    when the model performance does not improve for a given number of epochs (patience).

    Args:
        activation: Activation function applied to the teacher prediction.
        confidence_threshold: Threshold for computing a mask for filtering the pseudo labels.
            If none is given no mask will be computed.
        threshold_from_both_sides: Whether to include both values bigger than the threshold and smaller than 1 - it,
            or only values bigger than it in the mask. The former should be used for binary labels,
            the latter for for multiclass labels.
        mode: Determines whether the confidence threshold reduction is triggered by a "min" or "max" metric.
            - 'min': A lower value of the monitored metric is considered better (e.g., loss).
            - 'max': A higher value of the monitored metric is considered better (e.g., accuracy).
        factor Factor by which the confidence threshold is reduced when the performance stagnates.
        patience: Number of epochs (with no improvement) after which the confidence threshold will be reduced.
        threshold: Threshold value for determining a significant improvement in the performance metric
            to reset the patience counter. This can be relative (percentage improvement)
            or absolute depending on `threshold_mode`.
        threshold_mode: Determines whether the `threshold` is interpreted as a relative improvement ('rel')
            or an absolute improvement ('abs').
        min_ct: Minimum allowed confidence threshold. The threshold will not be reduced below this value.
        eps: A small value to avoid floating-point precision errors during threshold comparison.
        verbose: If True, prints messages when the confidence threshold is reduced.
    """

    def __init__(
        self,
        activation: Optional[torch.nn.Module] = None,
        confidence_threshold: Optional[float] = None,
        threshold_from_both_sides: bool = True,
        mode: Literal["min", "max"] = "min",
        factor: float = 0.05,
        patience: int = 10,
        threshold: float = 1e-4,
        threshold_mode: Literal["rel", "abs"] = "abs",
        min_ct: float = 0.5,
        eps: float = 1e-8,
        verbose: bool = True,
    ):
        super().__init__()
        self.activation = activation
        self.confidence_threshold = confidence_threshold
        self.threshold_from_both_sides = threshold_from_both_sides
        self.init_kwargs = {
            "activation": None,
            "confidence_threshold": confidence_threshold,
            "threshold_from_both_sides": threshold_from_both_sides,
        }
        # scheduler arguments
        if mode not in {"min", "max"}:
            raise ValueError(f"Invalid mode: {mode}. Mode should be 'min' or 'max'.")
        self.mode = mode

        if factor >= 1.0:
            raise ValueError("Factor should be < 1.0.")
        self.factor = factor

        self.patience = patience
        self.threshold = threshold

        if threshold_mode not in {"rel", "abs"}:
            raise ValueError(
                f"Invalid threshold mode: {mode}. Threshold mode should be 'rel' or 'abs'."
            )
        self.threshold_mode = threshold_mode

        self.min_ct = min_ct
        self.eps = eps
        self.verbose = verbose

        if mode == "min":
            self.best = float("inf")
        else:  # mode == "max":
            self.best = float("-inf")

        # self.best = 0
        self.num_bad_epochs: int = 0
        self.last_epoch = 0

    def _compute_label_mask_both_sides(
        self, pseudo_labels: torch.Tensor
    ) -> torch.Tensor:
        assert (
            self.confidence_threshold is not None
        ), "confidence_threshold must be set to compute label mask from both sides."
        upper_threshold = self.confidence_threshold
        lower_threshold = 1.0 - self.confidence_threshold
        mask = (
            (pseudo_labels >= upper_threshold) + (pseudo_labels <= lower_threshold)
        ).to(dtype=torch.float32)
        return mask

    def _compute_label_mask_one_side(self, pseudo_labels: torch.Tensor):
        mask = pseudo_labels >= self.confidence_threshold
        return mask

    def __call__(self, teacher: torch.nn.Module, input_: torch.Tensor):
        pseudo_labels = teacher(input_)
        if self.activation is not None:
            pseudo_labels = self.activation(pseudo_labels)
        if self.confidence_threshold is None:
            label_mask = None
        else:
            label_mask = (
                self._compute_label_mask_both_sides(pseudo_labels)
                if self.threshold_from_both_sides
                else self._compute_label_mask_one_side(pseudo_labels)
            )
        return pseudo_labels, label_mask

    def _is_better(self, a: float, best: float):
        if self.mode == "min" and self.threshold_mode == "rel":
            rel_epsilon = 1.0 - self.threshold
            return a < best * rel_epsilon

        elif self.mode == "min" and self.threshold_mode == "abs":
            return a < best - self.threshold

        elif self.mode == "max" and self.threshold_mode == "rel":
            rel_epsilon = self.threshold + 1.0
            return a > best * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + self.threshold

    def _reduce_ct(self, epoch: int):
        assert (
            self.confidence_threshold is not None
        ), "confidence_threshold must be set to reduce it."
        old_ct = self.confidence_threshold
        if self.threshold_mode == "rel":
            new_ct = max(self.confidence_threshold * self.factor, self.min_ct)
        else:  # threshold_mode == 'abs':
            new_ct = max(self.confidence_threshold - self.factor, self.min_ct)
        if old_ct - new_ct > self.eps:
            self.confidence_threshold = new_ct
        if self.verbose:
            print(
                f"Epoch {epoch}: reducing confidence threshold from {old_ct} to {self.confidence_threshold}"
            )

    def step(self, metric: Optional[float], epoch: Optional[int] = None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.last_epoch = epoch

        # If the metric is None, reduce the confidence threshold every epoch
        if metric is None:
            if epoch == 0:
                return
            if epoch % self.patience == 0:
                self._reduce_ct(epoch)
            return

        else:
            current = float(metric)

            if self._is_better(current, self.best):
                self.best = current
                self.num_bad_epochs = 0
            else:
                self.num_bad_epochs += 1

            if self.num_bad_epochs > self.patience:
                self._reduce_ct(epoch)
                self.num_bad_epochs = 0


class DummyDirectEvalPseudoLabeler:
    def __init__(
        self,
        score_threshold: Optional[float],
        activation: Optional[torch.nn.Module] = None,
    ):
        super().__init__()
        self.score_threshold = score_threshold
        self.consistency_log: List[float] = []
        self.activation = activation

    def _compute_label_mask(
        self, pseudo_labels: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        eval_scores = MultiClassF1Eval()(pseudo_labels, labels)
        mask = torch.zeros_like(pseudo_labels, dtype=torch.int8)
        # find ids where eval_scores is greater than score_threshold
        assert (
            self.score_threshold is not None
        ), "score_threshold must be set to compute label mask."
        ids = torch.argwhere(eval_scores[:, 1] > self.score_threshold).squeeze()
        mask[ids] = 1

        for score in eval_scores[:, 1]:
            self.consistency_log.append(score.item())

        return mask, eval_scores

    def __call__(
        self, teacher: torch.nn.Module, input_: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        raw_input_ = input_[:, 0:1, ...]  # Assuming input is of shape (B, C, D, H, W)
        label_gt_ = input_[:, 1:2, ...]  # Assuming input is of shape (B, C, D, H, W)
        pseudo_labels = teacher(raw_input_)
        assert is_torch_tensor(pseudo_labels), "pseudo_labels is not a torch.Tensor."

        if self.activation is not None:
            pseudo_labels = self.activation(pseudo_labels)

        if self.score_threshold is None:
            label_mask = None
        else:
            label_mask, _ = self._compute_label_mask(
                pseudo_labels,
                label_gt_,
            )
            label_mask = label_mask.to(input_.device)
        assert is_torch_tensor(pseudo_labels), (
            "pseudo_labels is not a torch.Tensor. "
            "Either pseudo_labels or label_mask must be a torch.Tensor."
        )
        return pseudo_labels, label_mask

    def step(self, metric: Optional[float], epoch: int):
        pass
