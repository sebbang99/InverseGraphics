from jaxtyping import Float
from omegaconf import DictConfig
from torch import Tensor

from .field import Field

import torch
import torch.nn as nn


class FieldGrid(Field):
    def __init__(
        self,
        cfg: DictConfig,
        d_coordinate: int,
        d_out: int,
    ) -> None:
        """Set up a grid for the neural field. Your architecture must respect the
        following parameters from the configuration (in config/field/grid.yaml):

        - side_length: the side length in each dimension

        Your architecture only needs to support 2D and 3D grids.
        """
        super().__init__(cfg, d_coordinate, d_out)
        assert d_coordinate in (2, 3)

        self.side_length = cfg.grid.side_length
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if d_coordinate == 2:
            self.grid = nn.Parameter(
                torch.randn(
                    (d_out, self.side_length, self.side_length),
                    dtype=torch.float32,
                    device=device,
                )
                * 0.01
            )
        elif d_coordinate == 3:
            self.grid = nn.Parameter(
                torch.randn(
                    (d_out, self.side_length, self.side_length, self.side_length),
                    dtype=torch.float32,
                    device=device,
                )
                * 0.01
            )

        # raise NotImplementedError("This is your homework.")

    def forward(
        self,
        coordinates: Float[Tensor, "batch coordinate_dim"],
    ) -> Float[Tensor, "batch output_dim"]:
        """Use torch.nn.functional.grid_sample to bilinearly sample from the image grid.
        Remember that your implementation must support either 2D and 3D queries,
        depending on what d_coordinate was during initialization.
        """

        normalized_coordinates = 2 * coordinates - 1  # to [-1, 1]
        # input_grid는 공간 구조. sampling 당할 대상.
        # grid는 sampling할 좌표.

        if self.d_coordinate == 2:
            # input: [1, C, H, W], grid: [1, B, 1, 2]
            input_grid = self.grid.unsqueeze(0)  # [1, C, H, W]
            grid = normalized_coordinates.unsqueeze(0).unsqueeze(2)  # [1, B, 1, 2]

            output = torch.nn.functional.grid_sample(
                input_grid,
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
            # output: [1, C, B, 1] → [B, C]
            return output.squeeze(0).squeeze(-1).permute(1, 0)  # [B, C]

        elif self.d_coordinate == 3:
            # input: [1, C, D, H, W], grid: [1, B, 1, 1, 3]
            input_grid = self.grid.unsqueeze(0)  # [1, C, D, H, W]
            grid = (
                normalized_coordinates.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            )  # [1, B, 1, 1, 3]

            # print(f"{input_grid.shape=}")
            # print(f"{grid.shape=}")
            output = torch.nn.functional.grid_sample(
                input_grid,
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )

            # output: [1, C, B, 1, 1] → [B, C]
            return output.squeeze(0).squeeze(-1).squeeze(-1).permute(1, 0)  # [B, C]

        raise NotImplementedError("This is your homework.")
