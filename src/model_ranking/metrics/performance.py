import torch
import torch.nn as nn
from typing import Optional, Literal

from torcheval.metrics.functional import (
    binary_f1_score,
    multiclass_f1_score,
    binary_accuracy,
)

from torch_em.loss.dice import (
    dice_score,
)

from ._utils import ensure_binary

from pytorch3dunet.unet3d.metrics import (
    DiceCoefficient,
)


class MultiClassF1Eval:
    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold

    def __call__(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        batch_perf_scores = torch.zeros((pred.shape[0], 2))
        # Loop over batch dimension
        for j in range(pred.shape[0]):
            pred_th = (pred[j] > self.threshold).type(torch.int64)
            if gt[j].sum() == 0:
                # Prevent warning from empty GT patches
                batch_perf_scores[j, 1] = 0
                batch_perf_scores[j, 0] = binary_f1_score(
                    (1 - pred[j]).flatten(),
                    (1 - gt[j].type(torch.int64)).flatten(),
                    threshold=self.threshold,
                )
            else:
                batch_perf_scores[j] = multiclass_f1_score(
                    pred_th.flatten(),
                    gt[j].flatten().type(torch.int64),
                    num_classes=2,
                    average=None,
                )
        return batch_perf_scores


class BinaryAccuracyEval:
    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold

    def __call__(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        batch_perf_scores = torch.zeros(pred.shape[0])
        # Loop over batch dimension
        for j in range(pred.shape[0]):
            batch_perf_scores[j] = binary_accuracy(
                pred[j].flatten(), gt[j].long().flatten(), threshold=self.threshold
            )
        return batch_perf_scores


class BinaryF1Eval:
    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold

    def __call__(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        batch_perf_scores = torch.zeros(pred.shape[0])
        # Loop over batch dimension
        for j in range(pred.shape[0]):
            if gt[j].sum() == 0:
                # Prevent warning from empty GT patches
                batch_perf_scores[j] = 0
            else:
                batch_perf_scores[j] = binary_f1_score(
                    pred[j].flatten(), gt[j].flatten(), threshold=self.threshold
                )
        return batch_perf_scores


class SoftF1Eval:
    def __call__(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        batch_perf_scores = torch.zeros(pred.shape[0])
        # Loop over batch dimension
        for j in range(pred.shape[0]):
            # add channel dimension to gt if needed
            if len(gt.shape) == 4:
                gt_patch = torch.unsqueeze(gt[j : j + 1], 1)
            else:
                gt_patch = gt[j : j + 1]
            batch_perf_scores[j] = DiceCoefficient()(pred[j : j + 1], gt_patch)
        return batch_perf_scores


class DiceMetric(nn.Module):
    """Metric computed based on the dice error between a binary input and binary target.

    Args:
        channelwise: Whether to return the dice score independently per channel.
        eps: The epsilon value added to the denominator for numerical stability.
        reduce_channel: How to return the dice score over the channel axis.
    """

    def __init__(
        self,
        channelwise: bool = True,
        eps: float = 1e-7,
        reduce_channel: Optional[str] = "sum",
        threshold: Optional[float] = None,
        final_activation: Optional[Literal["sigmoid", "softmax"]] = None,
    ):
        if reduce_channel not in ("sum", "mean", "max", "min", None):
            raise ValueError(f"Unsupported channel reduction {reduce_channel}")

        super().__init__()
        self.channelwise = channelwise
        self.eps = eps
        self.reduce_channel = reduce_channel
        self.threshold = threshold
        self.final_activation = final_activation

        # all torch_em classes should store init kwargs to easily recreate the init call
        self.init_kwargs = {
            "channelwise": channelwise,
            "eps": self.eps,
            "reduce_channel": self.reduce_channel,
        }

    def forward(self, input_: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the loss.

        Args:
            input_: The input, logits, probabilities or binary.
            target: The target, logits, probablities or binary.

        Returns:
            The dice score.
        """
        if self.final_activation is not None:
            if self.final_activation == "sigmoid":
                input_ = nn.functional.sigmoid(input_)
            elif self.final_activation == "softmax":
                input_ = nn.functional.softmax(input_, dim=1)
            else:
                raise ValueError(
                    f"Unsupported final activation {self.final_activation}"
                )

        if self.threshold is not None:
            input_ = ensure_binary(input_, threshold=self.threshold)
            target = ensure_binary(target, threshold=self.threshold)

        return dice_score(
            input_=input_,
            target=target,
            invert=False,
            channelwise=self.channelwise,
            eps=self.eps,
            reduce_channel=self.reduce_channel,
        )
