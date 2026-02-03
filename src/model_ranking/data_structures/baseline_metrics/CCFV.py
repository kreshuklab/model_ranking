from pydantic import BaseModel
from typing import Dict, List, Optional, Sequence

from model_ranking.data_structures.data.data import semantic_dataset_type
from model_ranking.data_structures.general.model import ModelSourceConfig


class CCFVFeatureConfig(BaseModel):
    layers: List[str]
    sample_num: Dict[str, int]
    num_classes: int


UNet_3Layers_CCFVConfig = CCFVFeatureConfig(
    layers=[
        "encoders.2",
        "decoders.0",
        "decoders.1",
    ],
    sample_num={
        "encoders.2": 200,
        "decoders.0": 400,
        "decoders.1": 800,
    },
    num_classes=1,
)


UNet_4Layers_CCFVConfig = CCFVFeatureConfig(
    layers=[
        "encoders.3",
        "decoders.0",
        "decoders.1",
        "decoders.2",
    ],
    sample_num={
        "encoders.3": 100,
        "decoders.0": 200,
        "decoders.1": 400,
        "decoders.2": 800,
    },
    num_classes=1,
)

ResUNet_Layers_CCFVConfig = CCFVFeatureConfig(
    layers=[
        "encoders.4",
        "decoders.0",
        "decoders.1",
        "decoders.2",
        "decoders.3",
    ],
    sample_num={
        "encoders.4": 25,
        "decoders.0": 50,
        "decoders.1": 100,
        "decoders.2": 200,
        "decoders.3": 400,
    },
    num_classes=1,
)

Unetr_Layers_CCFVConfig = CCFVFeatureConfig(
    layers=[
        "decoder5",
        "decoder4",
        "decoder3",
        "decoder2",
    ],
    sample_num={
        "decoder5": 100,
        "decoder4": 200,
        "decoder3": 400,
        "decoder2": 800,
    },
    num_classes=1,
)


class CCFVConfig(CCFVFeatureConfig):
    overwrite: bool
    save_path: str


class CCFVRunMetaConfig(BaseModel):
    target_datasets: Sequence[semantic_dataset_type]
    source_models: Sequence[ModelSourceConfig]
    overwrite_yaml: bool
    overwrite_scores: bool
    data_base_path: str
    model_dir_path: str
    model_key: str
    num_layers: Optional[int] = None
    output_base_path: str
