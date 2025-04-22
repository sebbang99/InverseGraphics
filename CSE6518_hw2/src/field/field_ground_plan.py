from jaxtyping import Float
from omegaconf import DictConfig
from torch import Tensor

from .field import Field

from .field_grid import FieldGrid
from .field_mlp import FieldMLP
from ..components.positional_encoding import PositionalEncoding
import torch


class FieldGroundPlan(Field):
    def __init__(
        self,
        cfg: DictConfig,
        d_coordinate: int,
        d_out: int,
    ) -> None:
        """Set up a neural ground plan. You should reuse the following components:

        - FieldGrid from  src/field/field_grid.py
        - FieldMLP from src/field/field_mlp.py
        - PositionalEncoding from src/components/positional_encoding.py

        Your ground plan only has to handle the 3D case.
        """
        super().__init__(cfg, d_coordinate, d_out)
        assert d_coordinate == 3

        self.grid_field = FieldGrid(
            cfg, d_coordinate=2, d_out=cfg.d_grid_feature
        )  # (x, y)
        self.z_encoder = PositionalEncoding(cfg.positional_encoding_octaves)  # z

        self.mlp_field = FieldMLP(
            cfg,
            d_coordinate=cfg.d_grid_feature + 2 * cfg.positional_encoding_octaves,
            d_out=d_out,
        )

        # raise NotImplementedError("This is your homework.")

    def forward(
        self,
        coordinates: Float[Tensor, "batch coordinate_dim"],
    ) -> Float[Tensor, "batch output_dim"]:
        """Evaluate the ground plan at the specified coordinates. You should:

        - Sample the grid using the X and Y coordinates.
        - Positionally encode the Z coordinates.
        - Concatenate the grid's outputs with the corresponding encoded Z values, then
          feed the result through the MLP.
        """

        xy = coordinates[:, :2]
        z = coordinates[:, 2:]

        grid_features = self.grid_field(xy)
        z_encoded = self.z_encoder(z)
        # print(f"grid_features.shape: {grid_features.shape}")
        # print(f"z_encoded.shape: {z_encoded.shape}")

        combined = torch.cat([grid_features, z_encoded], dim=-1)
        output = self.mlp_field(combined)

        return output
        raise NotImplementedError("This is your homework.")
