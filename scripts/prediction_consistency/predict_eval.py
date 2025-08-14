from pytorch3dunet.unet3d.config import (
    load_config,
)
from model_ranking import predict_eval

if __name__ == "__main__":
    cfg, _ = load_config()
    predict_eval(cfg)
