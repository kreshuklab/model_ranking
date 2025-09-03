import h5py  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
from pathlib import Path
import torch
from tqdm import tqdm
import typer

from model_ranking import (
    ClassificationPredictConfig,
    classification_prediction_evaluation,
    copy_classification_config,
    get_classification_transfer,
    get_classification_TTA_loaders,
    load_checkpoint_resnet,
    predict_with_features,
    ResNet,
)

from pytorch3dunet.unet3d.config import (
    load_config_direct,  # pyright: ignore[reportUnknownVariableType]
)


def main(
    config: Path = typer.Option(
        ..., help="Path to config yaml for difference image calculation"
    )
):
    cfg_data, _ = load_config_direct(config)
    cfg = ClassificationPredictConfig.model_validate(cfg_data)

    # get TTA dataloaders
    loaders = get_classification_TTA_loaders(cfg.loader)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: No GPU available, using CPU")
    model = ResNet(cfg.model)
    model = load_checkpoint_resnet(
        cfg.model.modelname,
        model=model,
        path=str(cfg.model.ckpt_path),
        location=device,
        model_key="ResNet",
    )
    model = model.eval()

    for aug, loader in tqdm(loaders.items()):
        print(f"Applying {aug} Test Time Augmentation")
        if aug == "None":
            aug_path = "None"
        else:
            aug_path = Path(aug.split("_")[0]) / aug.split("_")[-1]
        transfer_title = get_classification_transfer(
            cfg.model.modelname, str(cfg.loader.dataset.raw_path)
        )
        save_dir_path = (
            Path(cfg.output.save_dir_path)
            / transfer_title
            / cfg.model.modelname
            / aug_path
        )
        # check if save path exists if not create it
        if not save_dir_path.exists():
            save_dir_path.mkdir(parents=True, exist_ok=True)

        save_path = save_dir_path / "predictions.h5"

        copy_classification_config(old_path=config, save_path=save_path)

        # Run classification prediction
        model_output = predict_with_features(
            model,
            loader=loader,
            device=torch.device(device),
            feature_layers=cfg.model.feature_layers,  # pyright: ignore
        )

        evaluation_scores = classification_prediction_evaluation(
            (model_output["predictions"] > cfg.output.prediction_threshold).astype(
                np.uint8
            ),
            model_output["labels"],
        )

        with h5py.File(save_path, "w") as f:
            for output_key, value in model_output.items():
                if output_key == "features":
                    for k, v in model_output["features"].items():
                        _ = f.create_dataset(k, data=v)
                else:
                    _ = f.create_dataset(output_key, data=value)

            for metric, score in zip(
                ["accuracy_error", "precision", "recall", "f1_score"],
                evaluation_scores,
            ):
                _ = f.create_dataset(metric, data=score)


if __name__ == "__main__":
    typer.run(main)
