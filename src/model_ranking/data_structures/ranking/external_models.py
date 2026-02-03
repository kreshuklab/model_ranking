from pydantic import BaseModel

from .consistency_metrics import AdaptedRandErrorConsisConfig

from model_ranking.data_structures.general.summary import SummaryResultsConfig


class LoadTransformerPredictionsConfig(BaseModel):
    model_name: str
    TTA_key: str
    base_dir_path: str
    file_identifier: str


class ConsistencyPatchedTransformerConfig(BaseModel):
    predictions_perturbed: LoadTransformerPredictionsConfig
    predictions_unperturbed: LoadTransformerPredictionsConfig
    metric_config: AdaptedRandErrorConsisConfig
    summary: SummaryResultsConfig
