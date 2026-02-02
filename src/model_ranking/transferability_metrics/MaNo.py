from numpy.typing import ArrayLike
import torch
import torch.nn as nn


def uniform_cross_entropy(num_classes: int, ood_logits: ArrayLike):
    ood_logits = torch.tensor(ood_logits).squeeze(1)
    logits = ood_logits.cuda()
    targets = torch.ones((logits.shape[0], num_classes)).cuda() * (1 / num_classes)
    loss = nn.functional.cross_entropy(logits, targets)

    losses = torch.Tensor(loss)
    return torch.mean(losses)


def uniform_binary_cross_entropy(ood_logits: ArrayLike) -> torch.Tensor:
    """
    Binary cross entropy with uniform target (0.5 probability for each class).
    Works with single logit values for binary classification.

    Args:
        ood_logits: Array of logits, shape (N,) or (N, 1)

    Returns:
        Mean binary cross entropy loss
    """
    ood_logits = torch.tensor(ood_logits).squeeze()
    logits = ood_logits.cuda()
    # Uniform target for binary classification is 0.5
    targets = torch.ones(logits.shape[0]).cuda() * 0.5
    loss = nn.functional.binary_cross_entropy_with_logits(logits, targets)
    return loss


def MaNo_evaluate(norm_type: int, delta: float, ood_logits: ArrayLike) -> float:

    with torch.no_grad():
        outputs = torch.tensor(ood_logits)
        outputs = scaling_method(outputs, delta)
        score: torch.Tensor = torch.norm(outputs, p=norm_type) / (  # pyright: ignore
            (outputs.shape[0] * outputs.shape[1]) ** (1 / norm_type)
        )

    scores = score.numpy()  # pyright: ignore
    return scores.mean()  # pyright: ignore


def scaling_method(logits: torch.Tensor, delta: float):
    if delta > 5:
        outputs = torch.softmax(logits, dim=1)
    else:
        outputs = logits + 1 + logits**2 / 2
        # min_value = torch.min(outputs, 1, keepdim=True)[0].expand_as(outputs)

        # Remove min values to ensure all entries are positive. This is especially
        # needed when the approximation order is higher than 2.
        outputs = torch.clamp(outputs, min=1e-8)

        # Shift to ensure all values are positive
        # min_vals = outputs.min(dim=1, keepdim=True)[0]
        # outputs = outputs - min_vals

        outputs = nn.functional.normalize(outputs, dim=1, p=1)
    return outputs
