from pydantic import BaseModel
from typing import List, Literal, Optional, Union


class SEGConfig(BaseModel):
    methods: List[str]
    target: str
    result_dir: str
    base_path: str
    save_dir: str
    agree_ratios: List[float]
    radii: List[int]
    output_name: str
    task: Literal["nuclei", "cells"]
    sample_ids: Optional[Union[List[int], List[str]]] = None
