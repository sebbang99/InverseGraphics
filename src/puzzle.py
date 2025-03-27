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
    R = ext[..., :3, :3]
    T = ext[..., :3, 3:4]

    # 1. Find the look vector.
    look = np.zeros(3)
    leftVectors = []
    w2c = False
    for i in range(2):
        if np.dot(-T, R[:, i]) == 2:  # w2c, same direction
            look = R[:, i]
            leftVectors.extend([R[:, (i + 1) % 3], R[:, (i + 2) % 3]])
            w2c = True
            break
        if np.dot(-T, R[:, i]) == -2:  # w2c, opposite direction
            look = -R[:, i]
            leftVectors.extend([R[:, (i + 1) % 3], R[:, (i + 2) % 3]])
            w2c = True
            break
        if np.dot(-T, R[i, :]) == 2:  # c2w, same direction
            look = R[i, :]
            leftVectors.extend([R[:, (i + 1) % 3], R[:, (i + 2) % 3]])
            break
        if np.dot(-T, R[i, :]) == -2:  # c2w, opposite direction
            look = -R[i, :]
            leftVectors.extend([R[:, (i + 1) % 3], R[:, (i + 2) % 3]])
            break

    # 2. Find the right vector.
    worldY = np.array([0, 1, 0])
    rightApprox = np.cross(worldY, look)
    right = np.array([0, 0, 0])
    if abs(np.dot(rightApprox, leftVectors[0])) < abs(
        np.dot(rightApprox, leftVectors[1])
    ):
        right = leftVectors[0]
    else:
        right = leftVectors[1]
    if np.dot(np.cross(look, right), worldY) < 0:  # adjust direction
        right = -right

    # 3. Find the up vector.
    up = np.cross(look, right)

    # 4. Compose new extrinsic matrix.
    newR = np.column_stack((right, up, look))
    newT = T if w2c else -np.linalg.inv(R) @ R @ T
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
