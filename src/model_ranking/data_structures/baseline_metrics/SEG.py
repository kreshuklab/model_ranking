from pydantic import BaseModel
from typing import List, Literal


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
