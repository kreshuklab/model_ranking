from pydantic import BaseModel
from typing import Optional

from model_ranking.data_structures.data.utils import ForegroundFilterConfig


class SummaryResultsMetaConfig(BaseModel):
    overwrite_scores: bool
    consis_key: Optional[str] = None
    eval_key: Optional[str] = None
    save_name_postfix: str = ""
    filter_patches: bool = False


class SummaryResultsConfig(BaseModel):
    filter_patches: Optional[ForegroundFilterConfig]
    output_path: str
    eval_key: Optional[str]
    consis_key: Optional[str]
    overwrite_scores: bool
    save_name_postfix: str = ""
