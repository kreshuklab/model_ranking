import typer
from typing import Annotated

from model_ranking.dataclass import FeatureBasedTransferRankingConfig
from model_ranking.feature_ranking import FeatureBasedTransferRanking

from pytorch3dunet.unet3d.config import (
    load_config_direct,  # pyright: ignore[reportUnknownVariableType]
)


def main(
    config: Annotated[str, typer.Option(help="Path to the configuration file")],
):
    """
    Extract transfer features based on the provided configuration.
    """

    meta_cfg, _ = load_config_direct(config)
    feature_ranking_cfg = FeatureBasedTransferRankingConfig.model_validate(meta_cfg)

    # Load the configuration
    feature_ranking = FeatureBasedTransferRanking(feature_ranking_cfg)

    # Extract features
    feature_ranking.run_transfer_ranking_batched()


if __name__ == "__main__":
    typer.run(main)
