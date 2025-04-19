from jaxtyping import Float
from omegaconf import DictConfig
from torch import Tensor

from .field import Field

import torch.nn as nn
from ..components.positional_encoding import PositionalEncoding


class FieldMLP(Field):
    def __init__(
        self,
        cfg: DictConfig,
        d_coordinate: int,
        d_out: int,
    ) -> None:
        """Set up an MLP for the neural field. Your architecture must respect the
        following parameters from the configuration (in config/field/mlp.yaml):

        - positional_encoding_octaves: The number of octaves in the positional encoding.
          If this parameter is None, do not positionally encode the input.
        - num_hidden_layers: The number of hidden linear layers.
        - d_hidden: The dimensionality of the hidden layers.

        Don't forget to add ReLU between your linear layers!
        """

        super().__init__(cfg, d_coordinate, d_out)

        print(f"Config name: {cfg.name}")

        # config parameters
        positional_encoding_octaves = cfg.mlp.positional_encoding_octaves
        num_layers: int = cfg.mlp.num_hidden_layers
        d_hidden: int = cfg.mlp.d_hidden

        # Positional encoding
        if positional_encoding_octaves is not None:
            self.encoder = PositionalEncoding(positional_encoding_octaves)
            d_input = self.encoder.d_out(d_coordinate)
        else:
            self.encoder = None
            d_input = d_coordinate

        # set up an MLP
        layers = []

        # input layer
        layers.append(nn.Linear(d_input, d_hidden))
        layers.append(nn.ReLU())

        # hidden layers
        for _ in range(num_layers):
            layers.append(nn.Linear(d_hidden, d_hidden))
            layers.append(nn.ReLU())

        # output layer
        layers.append(nn.Linear(d_hidden, d_out))

        self.mlp = nn.Sequential(*layers)  # unpacking
        # raise NotImplementedError("This is your homework.")

    def forward(
        self,
        coordinates: Float[Tensor, "batch coordinate_dim"],
    ) -> Float[Tensor, "batch output_dim"]:
        """Evaluate the MLP at the specified coordinates."""

        if self.encoder is not None:
            coordinates = self.encoder(coordinates)

        return self.mlp(coordinates)  # mlp가 내부적으로 __call__()을 정의하고 있음.
        raise NotImplementedError("This is your homework.")
