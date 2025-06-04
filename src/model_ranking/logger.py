import os
from datetime import datetime
from PIL import Image
from typing import Optional, Literal
import torch
import numpy as np
import wandb
import torch_em  # pyright: ignore[reportMissingTypeStubs]

from model_ranking.utils import is_torch_tensor
import torch_em.transform  # pyright: ignore[reportMissingTypeStubs]
from torchvision.utils import make_grid  # pyright: ignore[reportMissingTypeStubs]
from torch_em.trainer.logger_base import (  # pyright: ignore[reportMissingTypeStubs]
    TorchEmLogger,
)
from torch_em.trainer.default_trainer import (  # pyright: ignore[reportMissingTypeStubs]
    DefaultTrainer,
)


def convert_to_pil_image(image: torch.Tensor):
    # image: (C, H, W)
    image = image.detach().cpu()
    if image.dtype != torch.uint8:
        image = (image * 255).clamp(0, 255).to(torch.uint8)
    image = image.permute(1, 2, 0)  # (H, W, C)
    np_img = image.numpy().astype(np.uint8)
    if np_img.shape[2] == 1:
        np_img = np_img.squeeze(2)  # (H, W)
    return Image.fromarray(np_img)


class SelfTrainingWandbLogger(TorchEmLogger):
    """Logger for self-training via `torch_em.self_training.FixMatch` or `torch_em.self_training.MeanTeacher`.

    Args:
        trainer: The instantiated trainer class.
        save_root: The root directory for saving the checkpoints and logs.
    """

    def __init__(
        self,
        trainer: DefaultTrainer,
        save_root: Optional[str],
        project_name: Optional[str] = None,
        log_model: Optional[Literal["gradients", "parameters", "all"]] = "all",
        log_model_freq: int = 1,
        log_model_graph: bool = True,
        mode: Literal["online", "offline", "disabled"] = "online",
        resume: Optional[str] = None,
    ):

        super().__init__(trainer, save_root)

        self.log_dir = (
            "./logs" if save_root is None else os.path.join(save_root, "logs")
        )
        os.makedirs(self.log_dir, exist_ok=True)

        self.wand_run = wandb.init(
            id=resume,
            project=project_name,
            name=trainer.name,
            dir=self.log_dir,
            mode=mode,
            resume="allow",
        )

        if trainer.name is None:
            if mode == "online":
                trainer.name = self.wand_run.name
            elif mode in ("offline", "disabled"):
                trainer.name = f"{mode}_{datetime.now():%Y-%m-%d_%H-%M-%S}"
            else:
                raise ValueError(mode)

        self.log_image_interval = trainer.log_image_interval

        wandb.watch(
            trainer.model,
            log=log_model,
            log_freq=log_model_freq,
            log_graph=log_model_graph,
        )

    def _add_supervised_images(
        self, step: int, name: str, x: torch.Tensor, y: torch.Tensor, pred: torch.Tensor
    ):
        if x.ndim == 5:
            assert y.ndim == pred.ndim == 5
            zindex = x.shape[2] // 2
            x, y, pred = x[:, :, zindex], y[:, :, zindex], pred[:, :, zindex]

        normalized_x = torch_em.transform.raw.normalize(  # pyright: ignore[reportUnknownVariableType]
            x[0]
        )
        assert is_torch_tensor(normalized_x), "Input tensor should be normalized"
        # Rearrange axis order from (z, y, x) to (y, x, z) by moving the first axis to the last
        grid = make_grid(
            [
                normalized_x,
                y[0, 0:1],
                pred[0, 0:1],
            ],
            padding=8,
        )

        wandb.log(
            {
                f"{name}/supervised/input-labels-prediction": [
                    wandb.Image(
                        convert_to_pil_image(grid),
                        caption="input-labels-prediction",
                    )
                ]
            },
            step=step,
        )

    def _add_unsupervised_images(
        self,
        step: int,
        name: str,
        x1: torch.Tensor,
        x2: torch.Tensor,
        pred: torch.Tensor,
        pseudo_labels: torch.Tensor,
        label_filter: Optional[torch.Tensor] = None,
    ):
        if x1.ndim == 5:
            assert x2.ndim == pred.ndim == pseudo_labels.ndim == 5
            zindex = x1.shape[2] // 2
            x1, x2, pred = x1[:, :, zindex], x2[:, :, zindex], pred[:, :, zindex]
            pseudo_labels = pseudo_labels[:, :, zindex]
            if label_filter is not None:
                assert label_filter.ndim == 5
                label_filter = label_filter[:, :, zindex]

        normalized_x1 = torch_em.transform.raw.normalize(  # pyright: ignore[reportUnknownVariableType]
            x1[0]
        )
        normalized_x2 = torch_em.transform.raw.normalize(  # pyright: ignore[reportUnknownVariableType]
            x2[0]
        )
        assert is_torch_tensor(normalized_x1) and is_torch_tensor(
            normalized_x2
        ), "Input tensors should be normalized"
        images = [
            normalized_x1,
            normalized_x2,
            pred[0, 0:1],
            pseudo_labels[0, 0:1],
        ]
        im_name = f"{name}/unsupervised/aug1-aug2-prediction-pseudolabels"
        if label_filter is not None:
            images.append(label_filter[0, 0:1])
            im_name += "-labelfilter"
        grid = make_grid(images, nrow=2, padding=8)

        wandb.log(
            {
                im_name: [
                    wandb.Image(
                        convert_to_pil_image(grid),
                        caption="aug1-aug2-prediction-pseudolabels",
                    )
                ]
            },
            step=step,
        )

    def log_combined_loss(self, step: int, loss: float):
        """@private"""
        wandb.log({"train/combined_loss": loss}, step=step)

    def log_lr(self, step: int, lr: float):
        """@private"""
        wandb.log({"train/learning_rate": lr}, step=step)

    def log_train_supervised(
        self,
        step: int,
        loss: float,
        x: torch.Tensor,
        y: torch.Tensor,
        pred: torch.Tensor,
    ):
        """@private"""
        wandb.log({"train/supervised/loss": loss}, step=step)
        if step % self.log_image_interval == 0:
            self._add_supervised_images(step, "validation", x, y, pred)

    def log_validation_supervised(
        self,
        step: int,
        metric: float,
        loss: float,
        x: torch.Tensor,
        y: torch.Tensor,
        pred: torch.Tensor,
    ):
        """@private"""
        wandb.log(
            {
                "validation/supervised/loss": loss,
                "validation/supervised/metric": metric,
            },
            step=step,
        )

        self._add_supervised_images(step, "validation", x, y, pred)

    def log_train_unsupervised(
        self,
        step: int,
        loss: float,
        x1: torch.Tensor,
        x2: torch.Tensor,
        pred: torch.Tensor,
        pseudo_labels: torch.Tensor,
        label_filter: Optional[torch.Tensor] = None,
    ):
        """@private"""
        wandb.log(
            {
                "train/unsupervised/loss": loss,
            },
            step=step,
        )
        if step % self.log_image_interval == 0:
            self._add_unsupervised_images(
                step, "train", x1, x2, pred, pseudo_labels, label_filter
            )

    def log_validation_unsupervised(
        self,
        step: int,
        metric: float,
        loss: float,
        x1: torch.Tensor,
        x2: torch.Tensor,
        pred: torch.Tensor,
        pseudo_labels: torch.Tensor,
        label_filter: Optional[torch.Tensor] = None,
    ):
        """@private"""
        wandb.log(
            {
                "validation/unsupervised/loss": loss,
                "validation/unsupervised/metric": metric,
            },
            step=step,
        )
        self._add_unsupervised_images(
            step, "validation", x1, x2, pred, pseudo_labels, label_filter
        )

    def log_ct(self, step: int, ct: float):
        wandb.log({"train/confidence_threshold": ct}, step=step)
