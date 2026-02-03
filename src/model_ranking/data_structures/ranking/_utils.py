from pydantic import BaseModel
from typing import List, Literal


class ForegroundRatioConfig(BaseModel):
    targets: List[str]
    model_names: List[str]
    run_id: str
    seg_key: str
    approach: Literal["consistency", "feature_perturbation_consistency"]
    base_path: str
    save_foreground_ratio: bool = True
    overwrite_ratios: bool = False


class ScaleConsistencyConfig(BaseModel):
    foreground_ratio_cfg: ForegroundRatioConfig
    consis_key: str
    summary_postfix: str = "_full"
    overwrite_scaled_consistency: bool = False
