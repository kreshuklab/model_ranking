from pydantic import BaseModel
from typing import Any, Dict, Optional

DEFAULT_SCHEDULER_KWARGS: Dict[str, Any] = {
    "mode": "max",
    "factor": 0.5,
    "patience": 10,
}


class SelfTrainingTrainConfig(BaseModel):
    lr: float
    n_iterations: Optional[int]
    epochs: Optional[int]
    save_ckpt_every_kth_epoch: Optional[int]
    mixed_precision: bool = True
    scheduler_kwargs: Dict[str, Any] = DEFAULT_SCHEDULER_KWARGS
    optimizer_kwargs: Dict[str, Any] = {}
