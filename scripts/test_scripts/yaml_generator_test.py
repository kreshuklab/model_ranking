from model_ranking.yaml_generators import generate_yaml

# from model_ranking.dataclass import Model3LayerSourceConfig
from pytorch3dunet.unet3d.config import load_config_direct  # type: ignore

config_path = "/g/kreshuk/talks/model_ranking/scripts/test_scripts/test_config3.yaml"

yaml_paths = generate_yaml(config_path)
print(yaml_paths)
