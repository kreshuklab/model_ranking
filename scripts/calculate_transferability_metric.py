from typing import Annotated
import typer

from model_ranking.dataclass import TransferabilityMetricConfig
from model_ranking.transfer_metrics import (
    transfer_sweep_transferability_metric,
)

from pytorch3dunet.unet3d.config import (
    load_config_direct,  # pyright: ignore[reportUnknownVariableType]
)


def main(
    config: Annotated[
        str,
        typer.Option(
            help="Path to the transferability metric config file", exists=True
        ),
    ],
):
    """
    Main function to run the transferability metric calculation.
    """
    cfg, _ = load_config_direct(config)
    transfer_cfg = TransferabilityMetricConfig.model_validate(cfg)
    _ = transfer_sweep_transferability_metric(transfer_cfg)


if __name__ == "__main__":
    typer.run(main)
