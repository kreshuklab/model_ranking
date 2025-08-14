import numpy as np
import torch
from tqdm import tqdm
from typing import Dict, Any

print("Starting script...")

try:
    from model_ranking import TransferFeatureExtraction

    print("Successfully imported TransferFeatureExtraction")
except Exception as e:
    print(f"Error importing TransferFeatureExtraction: {e}")
    import traceback

    traceback.print_exc()

try:
    from model_ranking import TransferFeatureExtractionConfig

    print("Successfully imported TransferFeatureExtractionConfig")
except Exception as e:
    print(f"Error importing TransferFeatureExtractionConfig: {e}")
    import traceback

    traceback.print_exc()

print("Script completed basic imports")
