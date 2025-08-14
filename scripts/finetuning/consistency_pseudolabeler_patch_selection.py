from pathlib import Path
import torch
from typing import List, Annotated
import typer
import numpy as np
import shutil
from tqdm import tqdm

# from model_ranking import get_unsupervised_loader
from model_ranking import (
    HmitoTargetConfig,
    MeanTeacherConfig,
    EPFLTargetConfig,
    RmitoTargetConfig,
    VNCTargetConfig,
)
from model_ranking.datasets import get_loaders
from model_ranking import (
    ModelConsistencyPatchWisePseudoLabeler,
    InputConsistencyPatchwisePseudoLabeler,
    DummyDirectEvalPseudoLabeler,
)
from model_ranking import find_transfer_from_pred_path

# from model_ranking import save_h5

# from model_ranking import is_torch_tensor
from pytorch3dunet.unet3d.model import (
    get_model,  # pyright: ignore[reportUnknownVariableType]
)
from pytorch3dunet.unet3d.utils import (
    load_checkpoint,  # pyright: ignore[reportUnknownVariableType]
)
from pytorch3dunet.unet3d.config import (
    load_config_direct,  # pyright: ignore[reportUnknownVariableType]
)
from pytorch3dunet.augment.transforms import (
    Transformer,
)


def main(
    config: Annotated[str, typer.Option(help="Path to the configuration file")],
    run_name: Annotated[str, typer.Option(help="Name of the run")],
):
    """
    Main function to run the consistency pseudolabeler patch selection.
    """

    cfg, _ = load_config_direct(config)
    MT_cfg = MeanTeacherConfig.model_validate(cfg["training"])
    checkpoint_dir_path = Path(MT_cfg.output_root_path) / "checkpoints" / MT_cfg.name
    for checkpoint in cfg["prediction"]["checkpoints"]:
        if checkpoint == "epoch-0":
            assert (
                MT_cfg.model_cfg.source_checkpoint is not None
            ), "Source checkpoint must be provided"
        else:
            MT_cfg.model_cfg.source_checkpoint = (
                checkpoint_dir_path / f"{checkpoint}.pt"
            )
        save_path = run_pseudolabeler_patch_selection(MT_cfg, run_name=run_name)
    assert isinstance(
        save_path, Path  # pyright: ignore[reportPossiblyUnboundVariable]
    ), "Save path should be a Path object"
    new_config_path = save_path.parent / "config.yaml"
    _ = shutil.copy2(config, new_config_path)


def run_pseudolabeler_patch_selection(config: MeanTeacherConfig, run_name: str):
    """
    Run the consistency pseudolabeler patch selection.
    """

    pseudo_labeler_config = config.pseudo_labeler_cfg

    if pseudo_labeler_config.activation is not None:
        if pseudo_labeler_config.activation == "softmax":
            activation = torch.nn.Softmax(dim=1)
        elif pseudo_labeler_config.activation == "sigmoid":
            activation = torch.nn.Sigmoid()
        else:
            raise ValueError(
                f"Unknown activation: {pseudo_labeler_config.activation}. "
                + "Supported are 'softmax' and 'sigmoid'."
            )
    else:
        activation = None

    if (pseudo_labeler_config.name == "input_consistency") or (
        pseudo_labeler_config.name == "model_consistency"
    ):
        consis_cfg = pseudo_labeler_config.consistency_metric
        if consis_cfg.name == "AdaptedRandError":
            consistency_metric = consis_cfg.initialise_metric(incomplete_gt=False)
        else:
            consistency_metric = consis_cfg.initialise_metric()

        # self training functionality
        if pseudo_labeler_config.name == "input_consistency":

            pseudo_labeler = InputConsistencyPatchwisePseudoLabeler(
                transformer=Transformer(
                    pseudo_labeler_config.transformer_cfg,
                    pseudo_labeler_config.stats_cfg,
                ),
                consistency_metric=consistency_metric,
                mask_threshold=consis_cfg.mask_threshold,
                consistency_threshold=pseudo_labeler_config.consistency_threshold,
                seg_params=pseudo_labeler_config.seg_params,
                activation=activation,
            )

        else:
            pseudo_labeler = ModelConsistencyPatchWisePseudoLabeler(
                perturbed_model_config=pseudo_labeler_config.perturbed_model_config,
                consistency_metric=consistency_metric,
                mask_threshold=consis_cfg.mask_threshold,
                consistency_threshold=pseudo_labeler_config.consistency_threshold,
                seg_params=pseudo_labeler_config.seg_params,
                activation=activation,
            )
    elif pseudo_labeler_config.name == "direct_eval_pseudo_labeler":
        pseudo_labeler = DummyDirectEvalPseudoLabeler(
            score_threshold=pseudo_labeler_config.score_threshold,
        )

    else:
        raise ValueError(
            f"Unsupported pseudo labeler: {pseudo_labeler_config.name}. "
            + "Supported are 'input_consistency', 'model_consistency', and 'direct_eval_pseudo_labeler'."
        )

    model_cfg = config.model_cfg
    model = get_model(model_cfg.model.model_dump())
    assert model_cfg.source_checkpoint is not None, "Source checkpoint must be provided"
    if not isinstance(model_cfg.source_checkpoint, Path):
        model_cfg.source_checkpoint = Path(model_cfg.source_checkpoint)
    if model_cfg.source_checkpoint.name.endswith(".pt"):
        model_key = "model_state"
    else:
        model_key = "model_state_dict"
    _ = load_checkpoint(model_cfg.source_checkpoint, model, model_key=model_key)
    model = model.to("cuda:0")
    model = model.eval()

    target = find_transfer_from_pred_path(config.output_root_path).split("_to_")[-1]

    # Set the appropriate target_config based on the target string
    if target == "EPFL":
        target_config = EPFLTargetConfig()
    elif target == "Hmito":
        target_config = HmitoTargetConfig()
    elif target == "Rmito":
        target_config = RmitoTargetConfig()
    elif target == "VNC":
        target_config = VNCTargetConfig()
    else:
        raise ValueError(f"Unknown target dataset: {target}")

    with torch.no_grad():
        for loader in get_loaders(
            target_config,
            phase="train",
            output_path=None,
            shuffle=False,
        ):
            # raw, _ = next(iter(loader))
            assert loader.batch_size is not None, "Batch size should not be None"
            all_one_ids: List[int] = []
            for i, (raw, label) in enumerate(tqdm(loader)):
                if isinstance(pseudo_labeler, DummyDirectEvalPseudoLabeler):
                    raw = torch.cat([raw, label], dim=1)

                raw = raw.to("cuda:0")
                _, label_mask = pseudo_labeler(model, raw)

                assert label_mask is not None, "Label mask should not be None"

                # Check if the spatial axis of label_mask are all 1, return the ids of all-one images
                # The spatial dimensions are the last 3 axes
                for j in range(label_mask.shape[0]):
                    if torch.all(label_mask[j, ...] == 1):
                        all_one_ids.append(i * loader.batch_size + j)

            assert (
                model_cfg.source_checkpoint is not None
            ), "Source checkpoint must be provided"
            if not isinstance(model_cfg.source_checkpoint, Path):
                model_cfg.source_checkpoint = Path(model_cfg.source_checkpoint)
            save_path = (
                Path(config.output_root_path)
                / "pseudo_labeler"
                / config.name
                / run_name
                / f"patch_selection_{model_cfg.source_checkpoint.stem}.npz"
            )

            save_path.parent.mkdir(parents=True, exist_ok=True)

            np.savez(
                save_path,
                accepted_ids=np.array(all_one_ids, dtype=int),
                consis_scores=np.array(pseudo_labeler.consistency_log, dtype=float),
            )

            # logging for debugging purposes
            # save_h5(
            #     save_path.parent / "pseudo_labels.h5",
            #     "pseudo_labels",
            #     np.concatenate(pseudo_labeler.log_pseudo_labels, axis=0),
            # )
            # save_h5(
            #     save_path.parent / "pseudo_labels.h5",
            #     "perturbed_pseudo_labels",
            #     np.concatenate(pseudo_labeler.log_pseudo_labels_perturbed, axis=0),
            # )
    assert isinstance(
        save_path, Path  # pyright: ignore[reportPossiblyUnboundVariable]
    ), "Save path should be a Path object"
    return save_path


if __name__ == "__main__":
    typer.run(main)
