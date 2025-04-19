from jaxtyping import Float
from omegaconf import DictConfig
from torch import Tensor, nn

from .field import Field

from ..components.sine_layer import SineLayer


class FieldSiren(Field):
    network: nn.Sequential

    def __init__(
        self,
        cfg: DictConfig,
        d_coordinate: int,
        d_out: int,
    ) -> None:
        """Set up a SIREN network using the sine layers at src/components/sine_layer.py.
        Your network should consist of:

        - An input sine layer whose output dimensionality is 256
        - Two hidden sine layers with width 256
        - An output linear layer
        """
        super().__init__(cfg, d_coordinate, d_out)

        # layers
        self.input_sine_layer = SineLayer(d_in=d_coordinate, d_out=256, is_first=True)
        self.hidden_sine_layer0 = SineLayer(d_in=256, d_out=256, omega_0=0.1)
        self.hidden_sine_layer1 = SineLayer(d_in=256, d_out=256, omega_0=0.1)
        self.output_layer = nn.Linear(256, d_out)

        print(self.input_sine_layer)

        # raise NotImplementedError("This is your homework.")

    def forward(
        self,
        coordinates: Float[Tensor, "batch coordinate_dim"],
    ) -> Float[Tensor, "batch output_dim"]:
        """Evaluate the MLP at the specified coordinates."""

        coordinates = (coordinates - 0.5) * 2  # [0, 1] to [-1, 1]

        x = self.input_sine_layer(coordinates)
        x = self.hidden_sine_layer0(x)
        x = self.hidden_sine_layer1(x)
        output = self.output_layer(x)

        return output
        raise NotImplementedError("This is your homework.")
