from pathlib import Path
from pydantic import BaseModel, Discriminator
from typing import Annotated, Optional, Union

from model_ranking.pydantic.general.model import (
    Pytorch3DUnetModelConfig,
    UnetrModelConfig,
    UnetrWithDropOutModelConfig,
)


class SelfTrainingModelConfig(BaseModel):
    model: Annotated[
        Union[Pytorch3DUnetModelConfig, UnetrModelConfig, UnetrWithDropOutModelConfig],
        Discriminator("name"),
    ]
    source_checkpoint: Optional[Union[str, Path]]
