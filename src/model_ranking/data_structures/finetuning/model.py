from pathlib import Path
from pydantic import BaseModel
from typing import Optional, Union

from model_ranking.data_structures.general.model import internal_model_type


class SelfTrainingModelConfig(BaseModel):
    model: internal_model_type
    source_checkpoint: Optional[Union[str, Path]]
