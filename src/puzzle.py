from pathlib import Path
from typing import Literal, TypedDict

from jaxtyping import Float
from torch import Tensor

import json
import numpy as np
from PIL import Image
import torch


class PuzzleDataset(TypedDict):
    extrinsics: Float[Tensor, "batch 4 4"]
    intrinsics: Float[Tensor, "batch 3 3"]
    images: Float[Tensor, "batch height width"]


def load_dataset(path: Path) -> PuzzleDataset:
    """Load the dataset into the required format."""

    metadata = json.loads((path / "metadata.json").open("r").read())

    image_names = path.glob("images/*.png")
    images = []
    for image_name in image_names:
        images.append(np.asarray(Image.open(image_name)))

    return PuzzleDataset(
        extrinsics=Tensor(metadata["extrinsics"]),
        intrinsics=Tensor(metadata["intrinsics"]),
        images=Tensor(np.array(images)),
    )
    raise NotImplementedError("This is your homework.")


def convert_dataset(dataset: PuzzleDataset) -> PuzzleDataset:
    """Convert the dataset into OpenCV-style camera-to-world format. As a reminder, this
    format has the following specification:

    - The camera look vector is +Z.
    - The camera up vector is -Y.
    - The camera right vector is +X.
    - The extrinsics are in camera-to-world format, meaning that they transform points
      in camera space to points in world space.

    """

    # 0. Get R and T from extrinsic matrix.
    ext = dataset["extrinsics"]
    R = ext[..., :3, :3]  # [32, 3, 3]
    T = ext[..., :3, 3:]  # [32, 3]

    # 1. Find the look vector.
    look = torch.zeros((R.shape[0], 3), dtype=torch.float32)  # [32, 3]
    leftVectors = []
    w2c = torch.zeros(R.shape[0], dtype=torch.bool)  # [32]

    for i in range(3):
        dotProduct = torch.einsum(
            "ij, ij -> i",
            torch.bmm(R.transpose(-1, -2), T).squeeze(),  # look vec
            R[:, i, :],  # camera axis
        )  # [32, 3] x [32, 3]
        print(f"{dotProduct=}")
        w2cMaskSame = torch.abs(dotProduct - 2) < 1e-4  # [32]
        print(f"{w2cMaskSame=}")
        if torch.any(w2cMaskSame):  # w2c, same direction
            look[w2cMaskSame] = R[w2cMaskSame, i, :]
            leftVectors.append(
                torch.cat(
                    [R[w2cMaskSame, (i + 1) % 3, :], R[w2cMaskSame, (i + 2) % 3, :]],
                    dim=1,
                )
            )
            w2c[w2cMaskSame] = True
            break

        w2cMaskOpposite = torch.abs(dotProduct + 2) < 1e-4
        print(f"{w2cMaskOpposite=}")
        if torch.any(w2cMaskOpposite):  # w2c, opposite direction
            look[w2cMaskOpposite] = -R[w2cMaskOpposite, i, :]
            leftVectors.append(
                torch.cat(
                    [
                        R[w2cMaskOpposite, (i + 1) % 3, :],
                        R[w2cMaskOpposite, (i + 2) % 3, :],
                    ],
                    dim=1,
                )
            )
            w2c[w2cMaskOpposite] = True
            break

        print(f"{T.shape=}")
        print(f"{R[:,:,0:1].shape=}")
        dotProduct = torch.einsum(
            "ij, ij -> i", -T.squeeze(), R[:, :, i : i + 1].squeeze()
        )  # [32, 3] x [32, 3]
        print(f"{dotProduct=}")
        c2wMaskSame = torch.abs(dotProduct - 2) < 1e-4
        print(c2wMaskSame)
        if torch.any(c2wMaskSame):  # c2w, same direction
            look[c2wMaskSame] = R[c2wMaskSame, :, i]
            leftVectors.append(
                torch.cat(
                    [R[c2wMaskSame, :, (i + 1) % 3], R[c2wMaskSame, :, (i + 2) % 3]],
                    dim=1,
                )
            )
            break

        c2wMaskOpposite = torch.abs(dotProduct + 2) < 1e-4
        print(c2wMaskOpposite)
        if torch.any(c2wMaskOpposite):  # c2w, opposite direction
            look[c2wMaskOpposite] = -R[c2wMaskOpposite, :, i]
            leftVectors.append(
                torch.cat(
                    [
                        R[c2wMaskOpposite, :, (i + 1) % 3],
                        R[c2wMaskOpposite, :, (i + 2) % 3],
                    ],
                    dim=1,
                )
            )
            break

    # 2. Find the right vector.
    worldY = torch.tensor([0, 1, 0], dtype=torch.float32).expand(32, -1)  # [32, 3]
    # print(worldY.shape)
    rightApprox = torch.cross(worldY, look)  # [32, 3]
    # print(rightApprox.shape)
    right = torch.zeros((R.shape[0], 3), dtype=torch.float32)

    # print(leftVectors.shape)
    # print(leftVectors[0].shape)
    cand0 = torch.abs(torch.sum(rightApprox * leftVectors[0][:, 0:3], dim=1))
    cand1 = torch.abs(torch.sum(rightApprox * leftVectors[0][:, 0:3], dim=1))
    right[cand0 < cand1] = leftVectors[0][:, 0:3][cand0 < cand1]
    right[cand0 >= cand1] = leftVectors[0][:, 0:3][cand0 >= cand1]

    lookCrossRight = torch.cross(look, right)  # adjust direction
    dotWorldY = torch.sum(lookCrossRight * worldY, dim=1)
    right[dotWorldY < 0] = -right[dotWorldY < 0]

    # 3. Find the up vector.
    up = torch.cross(look, right)  # [32, 3]
    # print(up.shape)

    # 4. Compose new extrinsic matrix.
    newR = torch.stack((right, up, look), dim=-1)
    print(w2c.shape)
    print(T.shape)
    print((-torch.linalg.inv(R) @ R @ T).shape)
    newT = torch.where(
        w2c.unsqueeze(1).expand(-1, 3),
        torch.bmm(R.transpose(-1, -2), T).squeeze(),
        T.squeeze(),
    ).unsqueeze(-1)
    print(newT.shape)
    ext[..., :3, :3] = newR
    ext[..., :3, -1:] = newT

    dataset["extrinsics"] = ext
    return dataset
    raise NotImplementedError("This is your homework.")


def quiz_question_1() -> Literal["w2c", "c2w"]:
    """In what format was your puzzle dataset?"""

    raise NotImplementedError("This is your homework.")


def quiz_question_2() -> Literal["+x", "-x", "+y", "-y", "+z", "-z"]:
    """In your puzzle dataset's format, what was the camera look vector?"""

    raise NotImplementedError("This is your homework.")


def quiz_question_3() -> Literal["+x", "-x", "+y", "-y", "+z", "-z"]:
    """In your puzzle dataset's format, what was the camera up vector?"""

    raise NotImplementedError("This is your homework.")


def quiz_question_4() -> Literal["+x", "-x", "+y", "-y", "+z", "-z"]:
    """In your puzzle dataset's format, what was the camera right vector?"""

    raise NotImplementedError("This is your homework.")


def explanation_of_problem_solving_process() -> str:
    """Please return a string (a few sentences) to describe how you solved the puzzle.
    We'll only grade you on whether you provide a descriptive answer, not on how you
    solved the puzzle (brute force, deduction, etc.).
    """

    raise NotImplementedError("This is your homework.")
