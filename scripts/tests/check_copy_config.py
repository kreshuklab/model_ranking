from model_ranking.config import copy_config
from pytorch3dunet.unet3d.config import (
    load_config_direct,  # pyright: ignore[reportUnknownVariableType]
)
from model_ranking.dataclass import MeanTeacherConfig

config_path = "/g/kreshuk/talks/model_ranking_results/Self-Finetuning/Mitochondria/test/config.yaml"

cfg, _ = load_config_direct(config_path)
mt_cfg = MeanTeacherConfig.model_validate(cfg)

copy_config(mt_cfg, config_path)
