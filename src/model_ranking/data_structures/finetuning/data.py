from pydantic import BaseModel
from typing import List, Optional, Sequence, Tuple


class DataNormalisationConfig(BaseModel):
    global_normalisation: bool = False
    global_percentiles: Optional[Tuple[float, float]] = None
    norm01: bool = False


class SelfTrainingDataConfig(BaseModel):
    unsupervised_train_paths: List[str]
    unsupervised_val_paths: List[str]
    patch_shape: Tuple[int, ...]
    raw_key: str
    label_key: Optional[str]
    batch_size: int
    num_workers: int
    n_samples_train: Optional[int]
    n_samples_val: Optional[int]
    roi_unsupervised_train: Optional[Sequence[Sequence[int]]] = None
    roi_unsupervised_val: Optional[Sequence[Sequence[int]]] = None
    normalisation: DataNormalisationConfig


class SupervisedDataConfig(BaseModel):
    patch_shape: Tuple[int, ...]
    supervised_train_paths: List[str]
    supervised_val_paths: List[str]
    raw_key: str
    label_key: str
    batch_size: int
    num_workers: int
    n_samples_train: Optional[int]
    n_samples_val: Optional[int]
    roi_supervised_train: Optional[Sequence[Sequence[int]]] = None
    roi_supervised_val: Optional[Sequence[Sequence[int]]] = None
