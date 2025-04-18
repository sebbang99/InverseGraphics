from jaxtyping import Float
from omegaconf import DictConfig
from torch import Tensor

from .field_dataset import FieldDataset

import torch
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F


class FieldDatasetImage(FieldDataset):
    def __init__(self, cfg: DictConfig) -> None:
        """Load the image in cfg.path into memory here."""

        print(cfg)

        super().__init__(cfg)

        image = Image.open(cfg.path).convert("RGB")

        transform = T.ToTensor()  # [0, 1]로 만들기
        self.image = transform(image)  # image를 자동으로 멤버 변수로 등록

        self.height, self.width = self.image.shape[1], self.image.shape[2]
        print(self.image.shape)  # shape (3, H, W)
        # raise NotImplementedError("This is your homework.")

    def query(
        self,
        coordinates: Float[Tensor, "batch d_coordinate"],
    ) -> Float[Tensor, "batch d_out"]:
        """Sample the image at the specified coordinates and return the corresponding
        colors. Remember that the coordinates will be in the range [0, 1].

        You may find the grid_sample function from torch.nn.functional helpful here.
        Pay special attention to grid_sample's expected input range for the grid
        parameter.
        """

        image = self.image.unsqueeze(0)  # image shape (1, 3, H, W)
        print(f"{image.shape=}")

        grid = coordinates * 2 - 1  # grid shape (4, 2)
        print(f"{grid.shape=}")

        grid = grid.view(1, -1, 1, 2)  # grid shape (1, 4, 1, 2)
        print(f"{grid.shape=}")

        sampled = F.grid_sample(
            image, grid, align_corners=True, mode="bilinear", padding_mode="border"
        )
        print(f"{sampled.shape=}")

        return sampled.squeeze(0).squeeze(2).permute(1, 0)
        raise NotImplementedError("This is your homework.")

    @property
    def d_coordinate(self) -> int:
        return 2

    @property
    def d_out(self) -> int:
        return 3

    @property
    def grid_size(self) -> tuple[int, ...]:
        """Return a grid size that corresponds to the image's shape."""

        raise NotImplementedError("This is your homework.")
