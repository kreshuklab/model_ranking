from pathlib import Path
import torch
from typing import Union

from model_ranking.data_structures import AdaptiveBatchNormConfig

from adabn.utils import (  # pyright: ignore[reportMissingTypeStubs]
    compute_bn_stats,  # pyright: ignore[reportUnknownVariableType]
    replace_bn_stats,  # pyright: ignore[reportUnknownVariableType]
    sequential_bn_adaptation,  # pyright: ignore[reportUnknownVariableType]
)
from pytorch3dunet.datasets.utils import (
    get_test_loaders,  # pyright: ignore[reportUnknownVariableType]
)
from pytorch3dunet.unet3d.model import (
    get_model,  # pyright: ignore[reportUnknownVariableType]
)
from pytorch3dunet.unet3d.utils import (
    load_checkpoint,  # pyright: ignore[reportUnknownVariableType]
)


def copy_checkpoint_with_updated_model(
    source_checkpoint_path: Union[str, Path],
    new_checkpoint_dir_path: Union[str, Path],
    updated_model: torch.nn.Module,
) -> None:
    """
    Copy a PyTorch checkpoint file and replace the model state dict with an updated model.

    Args:
        source_checkpoint_path: Path to the original checkpoint file
        new_checkpoint_path: Path where the new checkpoint should be saved
        updated_model: The model with updated batch normalization statistics
    """
    # Load the original checkpoint
    checkpoint = torch.load(source_checkpoint_path, map_location="cpu")

    # Determine the correct key for model state dict based on file extension
    source_path = Path(source_checkpoint_path)
    if source_path.suffix == ".pt":
        model_key = "model_state"
    else:
        model_key = "model_state_dict"

    # Get the updated model state dict
    if isinstance(updated_model, torch.nn.DataParallel):
        updated_state_dict = updated_model.module.state_dict()  # type: ignore
    else:
        updated_state_dict = updated_model.state_dict()

    # Replace the model state dict in the checkpoint
    checkpoint[model_key] = updated_state_dict

    # Create the directory for the new checkpoint if it doesn't exist
    new_checkpoint_dir = Path(new_checkpoint_dir_path)
    new_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    new_checkpoint_path = new_checkpoint_dir / source_path.name
    # Save the updated checkpoint to the new location
    torch.save(checkpoint, new_checkpoint_path)
    print(f"Updated checkpoint saved to: {new_checkpoint_path}")


def run_adaptive_batchnorm(config: AdaptiveBatchNormConfig):
    model_cfg = config.model_cfg
    model = get_model(model_cfg.model.model_dump())
    print("Initialize from source model:", model_cfg.source_checkpoint)
    src_ckpt_path = model_cfg.source_checkpoint
    assert src_ckpt_path is not None, "Source checkpoint must be provided"
    if Path(src_ckpt_path).suffix == ".pt":
        model_key = "model_state"
    else:
        model_key = "model_state_dict"
    _ = load_checkpoint(src_ckpt_path, model, model_key=model_key)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model = model.eval()

    test_loaders = list(get_test_loaders(config.model_dump()))
    assert (
        len(test_loaders) == 1
    ), f"Expected exactly one test loader, got {len(test_loaders)}"
    test_loader = test_loaders[0]

    print("Computing BatchNorm stats on test loader...")
    bn_stats = compute_bn_stats(  # pyright: ignore[reportUnknownVariableType]
        model, test_loader
    )
    print("Replacing BatchNorm stats in the model...")
    replace_bn_stats(model, bn_stats)

    return model


def run_sequential_adaptive_batchnorm(config: AdaptiveBatchNormConfig):
    model_cfg = config.model_cfg
    model = get_model(model_cfg.model.model_dump())
    print("Initialize from source model:", model_cfg.source_checkpoint)
    src_ckpt_path = model_cfg.source_checkpoint
    assert src_ckpt_path is not None, "Source checkpoint must be provided"
    if Path(src_ckpt_path).suffix == ".pt":
        model_key = "model_state"
    else:
        model_key = "model_state_dict"
    _ = load_checkpoint(src_ckpt_path, model, model_key=model_key)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model = model.eval()

    test_loaders = list(get_test_loaders(config.model_dump()))
    assert (
        len(test_loaders) == 1
    ), f"Expected exactly one test loader, got {len(test_loaders)}"
    test_loader = test_loaders[0]

    print("Computing BatchNorm stats on test loader...")
    _ = sequential_bn_adaptation(  # pyright: ignore[reportUnknownVariableType]
        model, test_loader, verbose=False
    )
    return model
