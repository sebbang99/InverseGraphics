from jaxtyping import Float
from torch import Tensor
import torch
from einops import einsum


def homogenize_points(
    points: Float[Tensor, "*batch dim"],
) -> Float[Tensor, "*batch dim+1"]:
    """Turn n-dimensional points into (n+1)-dimensional homogeneous points."""

    return torch.cat(
        [points, torch.ones(points.shape[0], 1, dtype=points.dtype)], dim=-1
    )
    raise NotImplementedError("This is your homework.")


def homogenize_vectors(
    points: Float[Tensor, "*batch dim"],
) -> Float[Tensor, "*batch dim+1"]:
    """Turn n-dimensional vectors into (n+1)-dimensional homogeneous vectors."""

    return torch.cat(
        [points, torch.zeros(points.shape[0], 1, dtype=points.dtype)], dim=-1
    )
    raise NotImplementedError("This is your homework.")


def transform_rigid(
    xyz: Float[Tensor, "*#batch 4"],
    transform: Float[Tensor, "*#batch 4 4"],
) -> Float[Tensor, "*batch 4"]:
    """Apply a rigid-body transform to homogeneous points or vectors."""

    raise NotImplementedError("This is your homework.")


def transform_world2cam(
    xyz: Float[Tensor, "*#batch 4"],
    cam2world: Float[Tensor, "*#batch 4 4"],
) -> Float[Tensor, "*batch 4"]:
    """Transform points or vectors from homogeneous 3D world coordinates to homogeneous
    3D camera coordinates.
    """

    world2cam = torch.linalg.inv(cam2world)
    return torch.einsum("...ij, ...j -> ...i", world2cam, xyz)
    raise NotImplementedError("This is your homework.")


def transform_cam2world(
    xyz: Float[Tensor, "*#batch 4"],
    cam2world: Float[Tensor, "*#batch 4 4"],
) -> Float[Tensor, "*batch 4"]:
    """Transform points or vectors from homogeneous 3D camera coordinates to homogeneous
    3D world coordinates.
    """

    raise NotImplementedError("This is your homework.")


def project(
    xyz: Float[Tensor, "*#batch 4"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
) -> Float[Tensor, "*batch 2"]:
    """Project homogenized 3D points in camera coordinates to pixel coordinates."""

    zero_col = torch.zeros(
        intrinsics.shape[0],
        intrinsics.shape[1],
        intrinsics.shape[2],
        1,
        dtype=torch.float32,
    )
    # print(zero_col.shape)
    # print(intrinsics.shape)
    # print(xyz.shape)

    intrinsics = torch.cat((intrinsics, zero_col), dim=3)
    print(intrinsics.shape)
    intrinsics = intrinsics.expand(xyz.shape[0], xyz.shape[1], -1, -1)
    print(intrinsics.shape)
    wdc_vertices = torch.einsum("...ij, ...j -> ...i", intrinsics, xyz)
    print(wdc_vertices.shape)
    return wdc_vertices[..., :-1] / wdc_vertices[..., -1:]
    raise NotImplementedError("This is your homework.")
