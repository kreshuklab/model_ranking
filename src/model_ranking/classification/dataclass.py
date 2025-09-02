from pathlib import Path
from pydantic import BaseModel
from typing import List, Optional, Union


class Conv1Config(BaseModel):
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int
    padding: int
    bias: bool


class ClassificationModelConfig(BaseModel):
    conv1: Conv1Config
    out_channels: int
    modelname: str
    ckpt_path: Union[str, Path]
    ckpt_key: str
    feature_layers: Optional[List[int]]
