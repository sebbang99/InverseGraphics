from jaxtyping import Float
from omegaconf import DictConfig
from torch import Tensor, nn

from .field.field import Field

import torch


class NeRF(nn.Module):
    cfg: DictConfig
    field: Field

    def __init__(self, cfg: DictConfig, field: Field) -> None:
        super().__init__()
        self.cfg = cfg
        self.field = field

    def forward(
        self,
        origins: Float[Tensor, "batch 3"],
        directions: Float[Tensor, "batch 3"],
        near: float,
        far: float,
    ) -> Float[Tensor, "batch 3"]:
        """Render the rays using volumetric rendering. Use the following steps:

        1. Generate sample locations along the rays using self.generate_samples().
        2. Evaluate the neural field at the sample locations. The neural field's output
           has four channels: three for RGB color and one for volumetric density. Don't
           forget to map these channels to valid output ranges.
        3. Compute the alpha values for the evaluated volumetric densities using
           self.compute_alpha_values().
        4. Composite these alpha values together with the evaluated colors from.
        """

        num_samples = self.cfg.get("num_samples", 64)  # 64는 기본 값.
        # print(f"{self.cfg=}")
        # print(f"{num_samples=}")

        # 1. Generate sample locations.
        points, boundaries = self.generate_samples(
            origins, directions, near, far, num_samples
        )

        # 2. Evaluate the neural field at the sample locations.
        outputs = self.field(points.view(-1, 3))
        outputs = outputs.view(origins.shape[0], num_samples, 4)
        emitted_radiance = torch.sigmoid(outputs[..., :3])
        density = torch.relu(
            outputs[..., 3]
        )  # [512, 128] : batch_size(ray 개수), num_samples(sample 개수)

        # 3. Density
        alphas = self.compute_alpha_values(density, boundaries)

        # 4. Color
        rgb = self.alpha_composite(alphas, emitted_radiance)

        return rgb
        raise NotImplementedError("This is your homework.")

    def generate_samples(
        self,
        origins: Float[Tensor, "batch 3"],
        directions: Float[Tensor, "batch 3"],
        near: float,
        far: float,
        num_samples: int,
    ) -> tuple[
        Float[Tensor, "batch sample 3"],  # xyz sample locations
        Float[Tensor, "batch sample+1"],  # sample boundaries
    ]:
        """For each ray, equally divide the space between the specified near and far
        planes into num_samples segments. Return the segment boundaries (including the
        endpoints at the near and far planes). Also return sample locations, which fall
        at the midpoints of the segments.
        """

        device = origins.device
        batch_size = origins.shape[0]  # 512

        # print(f"{batch_size=}")

        t_vals = torch.linspace(0.0, 1.0, steps=num_samples + 1, device=device)
        z_vals = near * (1.0 - t_vals) + far * t_vals
        z_vals = z_vals.expand(batch_size, num_samples + 1)

        z_mids = 0.5 * (
            z_vals[:, :-1] + z_vals[:, 1:]
        )  # [0 : n - 2]와 [1 : n - 1]을 더함.
        points = origins.unsqueeze(1) + directions.unsqueeze(1) * z_mids.unsqueeze(
            2
        )  # sample locations

        return points, z_vals
        raise NotImplementedError("This is your homework.")

    def compute_alpha_values(
        self,
        sigma: Float[Tensor, "batch sample"],
        boundaries: Float[Tensor, "batch sample+1"],
    ) -> Float[Tensor, "batch sample"]:
        """Compute alpha values from volumetric densities (values of sigma) and segment
        boundaries.
        """

        deltas = boundaries[:, 1:] - boundaries[:, :-1]
        alphas = 1.0 - torch.exp(-sigma * deltas)  # Beer-Lambert
        return alphas
        raise NotImplementedError("This is your homework.")

    def alpha_composite(
        self,
        alphas: Float[Tensor, "batch sample"],
        colors: Float[Tensor, "batch sample 3"],
    ) -> Float[Tensor, "batch 3"]:
        """Alpha-composite the supplied alpha values and colors. You may assume that the
        background is black.
        """

        T = torch.cumprod(
            torch.cat([torch.ones_like(alphas[:, :1]), 1.0 - alphas + 1e-10], dim=-1),
            dim=-1,
        )[:, :-1]
        weights = alphas * T

        rgb = torch.sum(weights.unsqueeze(-1) * colors, dim=1)
        return rgb
        raise NotImplementedError("This is your homework.")
