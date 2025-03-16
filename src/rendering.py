from jaxtyping import Float
from torch import Tensor
import torch
from src.geometry import *
from typing import cast


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
    homogenized_vertices = homogenize_points(vertices)
    print(homogenized_vertices.shape)

    homogenized_vertices = homogenized_vertices.unsqueeze(0)
    print(homogenized_vertices.shape)

    extrinsics = extrinsics.unsqueeze(1)
    ec_vertices = transform_world2cam(homogenized_vertices, extrinsics)
    print(ec_vertices.shape)

    # 3. Project the points onto the image plane.
    intrinsics = intrinsics.unsqueeze(1)
    wdc_vertices = project(ec_vertices, intrinsics)

    for i in range(wdc_vertices.shape[0]):
        for j in range(wdc_vertices.shape[1]):
            x, y = wdc_vertices[i, j]
            x, y = int(x * width), int(y * height)
            # print("x : " + str(x), ", y : " + str(y))
            if 0 <= x < white_canvas.shape[2] and 0 <= y < white_canvas.shape[1]:
                white_canvas[i, y, x] = 0.0

    return white_canvas
    raise NotImplementedError("This is your homework.")
