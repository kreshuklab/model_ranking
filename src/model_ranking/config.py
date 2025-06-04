from pathlib import Path
from typing import Union
import shutil

from model_ranking.dataclass import MeanTeacherConfig


def copy_config(config: MeanTeacherConfig, config_path: Union[str, Path]):
    if not isinstance(config_path, Path):
        config_path = Path(config_path)
    checkpoint_path = Path(config.output_root_path) / "checkpoints"
    assert checkpoint_path.exists(), f"Checkpoint path {checkpoint_path} does not exist"
    new_config_path = checkpoint_path / config.name / config_path.name
    new_config_path.parent.mkdir(parents=True, exist_ok=True)
    # Copy the config file to the new location
    _ = shutil.copy2(config_path, new_config_path)
