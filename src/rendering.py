from jaxtyping import Float
from torch import Tensor
import torch
from src.geometry import *


def render_point_cloud(
    vertices: Float[Tensor, "vertex 3"],
    extrinsics: Float[Tensor, "batch 4 4"],
    intrinsics: Float[Tensor, "batch 3 3"],
    resolution: tuple[int, int] = (256, 256),
) -> Float[Tensor, "batch height width"]:
    """Create a white canvas with the specified resolution. Then, transform the points
    into camera space, project them onto the image plane, and color the corresponding
    pixels on the canvas black.
    """

    # 1. Create a white canvas.
    height, width = resolution
    white_canvas = torch.ones((extrinsics.shape[0], height, width), dtype=torch.float32)

    # 2. Transform the points into camera space.
    # homogenized_vertices = homogenize_points(vertices)
    # ec_vertices = transform_world2cam(homogenized_vertices, extrinsics)
    return white_canvas

    raise NotImplementedError("This is your homework.")
