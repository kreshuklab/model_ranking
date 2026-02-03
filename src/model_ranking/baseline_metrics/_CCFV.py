import os
import numpy as np
from typing import Any, Dict

from CCFV.utils.evaluate import (  # pyright: ignore[reportMissingTypeStubs]
    evaluate_2d,  # pyright: ignore[reportUnknownVariableType]
)

from pytorch3dunet.datasets.utils import (
    get_val_loader,  # pyright: ignore[reportUnknownVariableType]
)
from pytorch3dunet.unet3d.model import (
    get_model,  # pyright: ignore[reportUnknownVariableType]
)
from pytorch3dunet.unet3d.utils import (
    load_checkpoint,  # pyright: ignore[reportUnknownVariableType]
)


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
