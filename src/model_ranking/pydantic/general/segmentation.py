from pydantic import BaseModel, Discriminator
from typing import Literal, Union, Annotated


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
