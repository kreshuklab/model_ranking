import typer
import os
from tqdm import tqdm

from model_ranking import (
    analyze_transfer_features,
    FeatureSpaceAnalysisConfig,
    get_source_from_model_name,
)

from pytorch3dunet.unet3d.config import (
    load_config_direct,  # pyright: ignore[reportUnknownVariableType]
)


def main(
    config: str = typer.Option(help="Path to config yaml", exists=True),
):
    meta_cfg, _ = load_config_direct(config)
    feature_analysis_cfg = FeatureSpaceAnalysisConfig.model_validate(meta_cfg)

    feature_cfg = feature_analysis_cfg.feature_config
    feature_ids = list(feature_cfg.layer_keys.keys())
    for target in feature_analysis_cfg.targets:
        for model_name in tqdm(feature_analysis_cfg.source_models):
            source = get_source_from_model_name(model_name)
            if (source == "VNC") and (target == "VNC"):
                continue
            else:
                model_identifier = model_name.split("_")[-1][:-1]
                if model_identifier in feature_ids:
                    key = feature_cfg.layer_keys[model_identifier]
                else:
                    key = "decoders.2"

            save_dir = os.path.join(
                feature_analysis_cfg.output_config.save_base_path,
                f"{model_name}_to_{target}",
                feature_analysis_cfg.output_config.save_name,
            )

            _ = analyze_transfer_features(
                model_name=model_name,
                target=target,
                layer_key=key,
                base_path=feature_cfg.base_path,
                output_dir=save_dir,
                max_samples=feature_analysis_cfg.max_samples,
            )


if __name__ == "__main__":
    typer.run(main)
