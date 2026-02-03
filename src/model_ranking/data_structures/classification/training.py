from pathlib import Path
from pydantic import BaseModel
from typing import Literal, Optional, Union

from .data import ClassificationLoaderConfig
from .model import ClassificationModelConfig

from model_ranking.data_structures.general import WandbConfig


class SchedulerConfig(BaseModel):
    mode: str = "min"
    factor: float = 0.5
    patience: int = 5


class LoggingSettings(BaseModel):
    log_image_interval: int
    log_val_images: bool
    log_pred: bool


class TrainingSettingsConfig(BaseModel):
    save_path: str
    num_epochs: int
    logging: LoggingSettings
    loss_function: Literal["BCEWithLogitsLoss"] = "BCEWithLogitsLoss"
    learning_rate: float = 1e-4
    scheduler_kwargs: SchedulerConfig = SchedulerConfig()


class ClassificationTrainConfig(BaseModel):
    wandb: WandbConfig
    train_loader: ClassificationLoaderConfig
    val_loader: ClassificationLoaderConfig
    test_loader: Optional[ClassificationLoaderConfig]
    model_cfg: ClassificationModelConfig
    training_config: TrainingSettingsConfig


class ClassificationOutputConfig(BaseModel):
    save_dir_path: Union[str, Path]
    prediction_threshold: float


class ClassificationPredictConfig(BaseModel):
    loader: ClassificationLoaderConfig
    model: ClassificationModelConfig
    output: ClassificationOutputConfig
