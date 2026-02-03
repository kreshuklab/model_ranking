from pydantic import BaseModel
from typing import Dict, List, Optional, Sequence

from model_ranking.pydantic.data.data import semantic_dataset_type
from model_ranking.pydantic.general.model import ModelSourceConfig


class FeatureSampleConfig(BaseModel):
    layers: List[str]
    sampling_seed: int
    num_samples: int
    output_dir_path: Optional[str]


class TransferFeatureExtractionConfig(BaseModel):
    target_datasets: Sequence[semantic_dataset_type]
    source_models: Sequence[ModelSourceConfig]
    source_model_base_path: str
    data_base_path: str
    feature_cfg: FeatureSampleConfig


class PrecomputedFeatureConfig(BaseModel):
    base_path: str
    file_type: str
    layer_keys: Dict[str, str]
    n_PCA_components: Optional[int] = None
