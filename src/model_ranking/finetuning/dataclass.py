from pydantic import BaseModel, Discriminator
from typing import Annotated, Union

from model_ranking.dataclass import (
    SelfTrainingModelConfig,
    Pytorch3DUnetTestLoaderConfig,
    SBIAD1410LoaderTestConfig,
    TIFPredictionLoadersConfig,
)

test_loaders_type = Annotated[
    Union[
        Pytorch3DUnetTestLoaderConfig,
        SBIAD1410LoaderTestConfig,
        TIFPredictionLoadersConfig,
    ],
    Discriminator("dataset"),
]


class AdaptiveBatchNormConfig(BaseModel):
    model_cfg: SelfTrainingModelConfig
    loaders: test_loaders_type
    output_checkpoint_dir_path: str
