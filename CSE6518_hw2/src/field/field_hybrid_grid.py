from jaxtyping import Float
from omegaconf import DictConfig
from torch import Tensor

from .field import Field
from .field_grid import FieldGrid
from .field_mlp import FieldMLP


class FieldHybridGrid(Field):
    def __init__(
        self,
        cfg: DictConfig,
        d_coordinate: int,
        d_out: int,
    ) -> None:
        """Set up a hybrid grid-mlp neural field. You should reuse FieldGrid from
        src/field/field_grid.py and FieldMLP from src/field/field_mlp.py in your
        implementation.

        Hint: Since you're reusing existing components, you only need to add one line
        each to __init__ and forward!
        """
        super().__init__(cfg, d_coordinate, d_out)

        self.grid_field = FieldGrid(cfg.grid, d_coordinate, cfg.d_grid_feature)
        self.mlp_field = FieldMLP(cfg.mlp, cfg.d_grid_feature, d_out)
        # raise NotImplementedError("This is your homework.")

    def forward(
        self,
        coordinates: Float[Tensor, "batch coordinate_dim"],
    ) -> Float[Tensor, "batch output_dim"]:

        grid_values = self.grid_field(coordinates)  # grid에서 값을 sampling함.
        # print(f"{grid_values.shape=}")
        output = self.mlp_field(grid_values)  # sampling한 값을 MLP에 통과시킴.
        # print(f"{output.shape=}")

        return output
        raise NotImplementedError("This is your homework.")
