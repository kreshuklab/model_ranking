import numpy as np
from numpy.typing import NDArray
from pydantic import Discriminator
import torch
from typing import Optional, Union, Any, Tuple, Annotated

from model_ranking.dataclass import (
    Pytorch3DUnetModelConfig,
    SemanticSegmentation,
    InstanceSegmentation,
)
from model_ranking.metrics import (
    get_mask,
    CrossEntropyEval,
    DifferenceImageEval,
    EffectiveInvarianceEval,
    EntropyEval,
    HammingDistanceEval,
    KLDivergenceEval,
    AdaptedRandErrorEval,
)
from model_ranking.utils import (
    is_torch_tensor,
)

from pytorch3dunet.augment.transforms import (
    RandomContrast,
    RandomGamma,
    RandomBrightness,
    AdditiveGaussianNoise,
    Transformer,
)
from pytorch3dunet.unet3d.model import (
    get_model,  # pyright: ignore[reportUnknownVariableType]
)
from pytorch3dunet.unet3d.predictor import (
    pmaps_to_IN_seg,  # pyright: ignore[reportUnknownVariableType]
)

consistency_metric_type = Union[
    CrossEntropyEval,
    DifferenceImageEval,
    EffectiveInvarianceEval,
    EntropyEval,
    HammingDistanceEval,
    KLDivergenceEval,
    AdaptedRandErrorEval,
]

transform_type = Union[
    RandomContrast,
    RandomGamma,
    RandomBrightness,
    AdditiveGaussianNoise,
]

segmentation_type = Annotated[
    Union[SemanticSegmentation, InstanceSegmentation],
    Discriminator("name"),
]


class AbstractConsistencyPatchwisePseudoLabeler:
    """Compute pseudo labels based on model predictions, typically from a teacher model.

    Args:
        consistency_threshold: Threshold for accepting patches, if the patch consistency
            is above the threshold the full prediction will be used, otherwise the patch will
            be masked out.
        consistency_metric: Metric used to compute the consistency of the patches.
        foreground_threhold: threshold to consider foreground only pixels for consistency
            calculation.
        seg_params: Segmentation parameters for processing prediction.

    """

    def __init__(
        self,
        consistency_metric: consistency_metric_type,
        foreground_threshold: Optional[float] = None,
        consistency_threshold: Optional[float] = None,
        seg_params: segmentation_type = SemanticSegmentation(),
    ):
        super().__init__()
        self.consistency_metric = consistency_metric
        self.foreground_threshold = foreground_threshold
        self.consistency_threshold = consistency_threshold
        self.seg_params = seg_params
        # TODO serialize the class names and kwargs for activation instead

    def _compute_label_mask(
        self, pseudo_labels: NDArray[Any], perturbed_pseudo_labels: NDArray[Any]
    ) -> torch.Tensor:
        if isinstance(self.consistency_metric, AdaptedRandErrorEval):
            consis_score, consis_mask = self.consistency_metric(
                perturbed_pseudo_labels, pseudo_labels
            )
        else:
            if self.foreground_threshold is None:
                consis_mask = np.ones_like(pseudo_labels)
            else:
                consis_mask = get_mask(
                    pseudo_labels, perturbed_pseudo_labels, self.foreground_threshold
                )
            consis_score = self.consistency_metric(
                perturbed_pseudo_labels, pseudo_labels, consis_mask
            )
        mask = np.zeros_like(pseudo_labels)
        if isinstance(self.consistency_metric, HammingDistanceEval):
            # Hamming distance is a measure of dissimilarity, so we want to keep
            # the patches with a low distance
            ids = np.argwhere(consis_score < self.consistency_threshold)

        elif isinstance(self.consistency_metric, AdaptedRandErrorEval):
            ids = np.argwhere(consis_score[:, 0] > self.consistency_threshold)

        else:
            consis_score_PP = np.array(
                np.nanmean(consis_score, axis=tuple(range(1, consis_score.ndim)))
            )
            ids = np.argwhere(consis_score_PP > self.consistency_threshold)

        mask[ids] = 1

        return torch.from_numpy(mask)

    def get_instance_labels(
        self, pseudo_labels: torch.Tensor, pseudo_labels_perturbed: torch.Tensor
    ):
        assert isinstance(self.seg_params, InstanceSegmentation), (
            "Segmentation type is not instance segmentation. "
            "Please use the correct segmentation type."
        )
        ps_labels = torch.zeros_like(pseudo_labels)
        ps_labels_perturbed = torch.zeros_like(pseudo_labels_perturbed)
        for i in range(len(pseudo_labels)):
            ps_labels[i] = (
                torch.from_numpy(
                    pmaps_to_IN_seg(
                        pseudo_labels[i].cpu().numpy().squeeze(),
                        min_size=self.seg_params.min_size,
                        zero_largest_instance=self.seg_params.zero_largest_instance,
                        no_adjust_background=self.seg_params.no_adjust_background,
                    )
                )
                .to(pseudo_labels.dtype)
                .to(pseudo_labels.device)
            )
            ps_labels_perturbed[i] = (
                torch.from_numpy(
                    pmaps_to_IN_seg(
                        pseudo_labels_perturbed[i].cpu().numpy().squeeze(),
                        min_size=self.seg_params.min_size,
                        zero_largest_instance=self.seg_params.zero_largest_instance,
                        no_adjust_background=self.seg_params.no_adjust_background,
                    )
                )
                .to(ps_labels_perturbed.dtype)
                .to(ps_labels_perturbed.device)
            )
        return ps_labels, ps_labels_perturbed


class InputConsistencyPatchwisePseudoLabeler(AbstractConsistencyPatchwisePseudoLabeler):
    """Compute pseudo labels based on model predictions, typically from a teacher model.
    Optionally apply patch filter depending on the model's predictions consistency under
    input space perturbations.

    Args:
        transformer: Transformer to apply to the input during consistency analysis.
        consistency_threshold: Threshold for accepting patches, if the patch consistency
            is above the threshold the full prediction will be used, otherwise the patch will
            be masked out.
        foreground_threshold: threshold to consider foreground only pixels for consistency
            calculation.
        consistency_metric: Metric used to compute the consistency of the patches.
        seg_params: Segmentation parameters for processing prediction.

    """

    def __init__(
        self,
        transformer: Transformer,
        consistency_metric: consistency_metric_type,
        foreground_threshold: Optional[float] = None,
        consistency_threshold: Optional[float] = None,
        seg_params: segmentation_type = SemanticSegmentation(),
    ):
        super().__init__(
            consistency_metric=consistency_metric,
            foreground_threshold=foreground_threshold,
            consistency_threshold=consistency_threshold,
            seg_params=seg_params,
        )
        self.transform = transformer.raw_transform()

    def __call__(
        self, teacher: torch.nn.Module, input_: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        pseudo_labels = teacher(input_)
        perturbed_input_ = (
            torch.from_numpy(self.transform(input_.cpu().numpy().astype("float32")))
            .to(input_.dtype)
            .to(input_.device)
        )
        pseudo_labels_perturbed = teacher(perturbed_input_)
        assert is_torch_tensor(pseudo_labels), "pseudo_labels is not a torch.Tensor."
        assert is_torch_tensor(
            pseudo_labels_perturbed
        ), "pseudo_labels_perturbed is not a torch.Tensor."
        if self.seg_params.name == "instance":
            pseudo_labels, pseudo_labels_perturbed = self.get_instance_labels(
                pseudo_labels, pseudo_labels_perturbed
            )

        if self.consistency_threshold is None:
            label_mask = None
        else:
            label_mask = self._compute_label_mask(
                pseudo_labels.cpu().numpy().astype("float32"),
                pseudo_labels_perturbed.cpu().numpy().astype("float32"),
            ).to(input_.device)
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
        foreground_threhold: threshold to consider foreground only pixels for
            consistency calculation.
        consistency_threshold: Threshold for accepting patches, if the patch consistency
            is above the threshold the full prediction will be used, otherwise the patch
            will be masked out.
        seg_params: Segmentation parameters for processing prediction.
    """

    def __init__(
        self,
        perturbed_model_config: Pytorch3DUnetModelConfig,
        consistency_metric: consistency_metric_type,
        foreground_threshold: Optional[float] = None,
        consistency_threshold: Optional[float] = None,
        seg_params: segmentation_type = SemanticSegmentation(),
    ):
        super().__init__(
            consistency_metric=consistency_metric,
            foreground_threshold=foreground_threshold,
            consistency_threshold=consistency_threshold,
            seg_params=seg_params,
        )
        self.perturbed_teacher = get_model(perturbed_model_config.model_dump())

    def __call__(
        self, teacher: torch.nn.Module, input_: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        pseudo_labels = teacher(input_)
        _ = self.perturbed_teacher.load_state_dict(teacher.state_dict())
        perturbed_teacher = self.perturbed_teacher.to(next(teacher.parameters()).device)
        perturbed_teacher = perturbed_teacher.eval()
        pseudo_labels_perturbed = perturbed_teacher(input_)

        if self.seg_params.name == "instance":
            pseudo_labels, pseudo_labels_perturbed = self.get_instance_labels(
                pseudo_labels, pseudo_labels_perturbed
            )

        if self.consistency_threshold is None:
            label_mask = None
        else:
            label_mask = self._compute_label_mask(
                pseudo_labels.cpu().numpy().astype("float32"),
                pseudo_labels_perturbed.cpu().numpy().astype("float32"),
            )
        assert is_torch_tensor(pseudo_labels), (
            "pseudo_labels is not a torch.Tensor. "
            "Either pseudo_labels or label_mask must be a torch.Tensor."
        )
        return pseudo_labels, label_mask
