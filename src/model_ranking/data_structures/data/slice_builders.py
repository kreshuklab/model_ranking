from pydantic import BaseModel
from typing import Literal, Tuple, Optional, Union, List


class Pytorch3DUnetSliceBuilderConfig(BaseModel):
    name: Literal["SliceBuilder"]
    patch_shape: Tuple[int, int, int]
    stride_shape: Tuple[int, int, int]
    halo_shape: Tuple[int, int, int]


class Pytorch3DUnetSingleZSliceBuilderConfig(BaseModel):
    name: Literal["SingleZSliceBuilder"]
    patch_shape: Tuple[int, int, int]
    stride_shape: Tuple[int, int, int]
    halo_shape: Tuple[int, int, int]


class Pytorch3DUnetFilterSliceBuilderConfig(BaseModel):
    name: Literal["FilterSliceBuilder"]
    patch_shape: Optional[Tuple[int, int, int]] = None
    stride_shape: Optional[Tuple[int, int, int]] = None
    halo_shape: Optional[Tuple[int, int, int]] = None
    threshold: float
    ignore_index: Optional[Optional[int]] = None
    slack_acceptance: float

class Pytorch3DUnetDistributionSliceBuilder(BaseModel):
    name: Literal["DistributionSliceBuilder"]
    patch_shape: Tuple[int, int, int]
    stride_shape: Tuple[int, int, int]
    halo_shape: Optional[Tuple[int, int, int]] = None
    bin_edges: List[float]
    target_distribution: List[float]
    ignore_index: Optional[Union[int, List[int]]] = None
    n_patches: Optional[int] = None 
    seed: int = 47
    save_dir: Optional[str] = None
    phase: Optional[str] = None


slice_builder_type = Union[
    Pytorch3DUnetSliceBuilderConfig,
    Pytorch3DUnetFilterSliceBuilderConfig,
    Pytorch3DUnetSingleZSliceBuilderConfig,
    Pytorch3DUnetDistributionSliceBuilder,
]
