import numpy as np
import torch
from tqdm import tqdm
from typing import Dict, Any

print("Starting script...")

try:
    from model_ranking.feature_ranking import FeatureBasedTransferRanking

    print("Successfully imported FeatureBasedTransferRanking")
except Exception as e:
    print(f"Error importing FeatureBasedTransferRanking: {e}")
    import traceback

    traceback.print_exc()

try:
    from model_ranking.dataclass import FeatureBasedTransferRankingConfig

    print("Successfully imported FeatureBasedTransferRankingConfig")
except Exception as e:
    print(f"Error importing FeatureBasedTransferRankingConfig: {e}")
    import traceback

    traceback.print_exc()

print("Script completed basic imports")
