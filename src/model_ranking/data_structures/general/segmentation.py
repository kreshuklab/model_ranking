from pydantic import BaseModel, Discriminator
from typing import List, Literal, Union, Annotated


class SemanticSegmentationConfig(BaseModel):
    name: Literal["semantic"] = "semantic"


class InstanceSegmentationConfig(BaseModel):
    name: Literal["instance"] = "instance"
    min_size: int = 50
    beta: float = 0.5
    zero_largest_instance: bool = False
    zero_large_instances: bool = False
    large_instance_multiplier: int = 4


segmentation_type = Annotated[
    Union[SemanticSegmentationConfig, InstanceSegmentationConfig],
    Discriminator("name"),
]


class CalculateSegmentationConfig(BaseModel):
    pred_base_path: str
    pred_key: str = "predictions"
    models: List[str]
    output_key: str = "segmentations"
    overwrite_output: bool = False
    min_size: int = 50
    zero_largest_instance: bool = False
    zero_large_instances: bool = True
    large_instance_multiplier: float = 1.7
    beta: float = 0.5
    max_obj_size: int = 5867
