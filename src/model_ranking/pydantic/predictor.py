from pydantic import BaseModel
from typing import Literal, Optional


class Pytorch3DUnetPredictorMetaConfig(BaseModel):
    name: Literal["PatchWisePredictor"]
    save_segmentation: bool
    min_size: Optional[int]
    layer_id: Optional[int]
    beta: float = 0.5
    zero_largest_instance: bool
    zero_large_instances: bool
    large_instance_multiplier: float = 4
    max_obj_size: Optional[float] = None
    threshold: float = 0.5


class Pytorch3DUnetPredictorConfig(Pytorch3DUnetPredictorMetaConfig):
    save_suffix: str
    output_file_name: Optional[str]


class TIFNucleiSemanticPredictorConfig(BaseModel):
    name: Literal["DSB2018Predictor"]


class TIFNucleiInstancePredictorConfig(BaseModel):
    name: Literal["NucleiInstancePredictor"]
    save_segmentation: bool
    min_size: int
    zero_largest_instance: bool = False
    no_adjust_background: bool = False
