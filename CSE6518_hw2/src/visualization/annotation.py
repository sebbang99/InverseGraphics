from pathlib import Path
from string import ascii_letters, digits, punctuation

import numpy as np
import torch
from einops import rearrange
from jaxtyping import Float
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor

from .layout import vcat

EXPECTED_CHARACTERS = digits + punctuation + ascii_letters


def draw_label(
    text: str,
    font: Path,
    font_size: int,
    device: torch.device = torch.device("cpu"),
) -> Float[Tensor, "3 height width"]:
    """Draw a black label on a white background with no border."""
    font = ImageFont.truetype(str(font), font_size)
    left, _, right, _ = font.getbbox(text)
    width = right - left
    _, top, _, bottom = font.getbbox(EXPECTED_CHARACTERS)
    height = bottom - top
    image = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(image)
    draw.text((0, 0), text, font=font, fill="black")
    image = torch.tensor(np.array(image) / 255, dtype=torch.float32, device=device)
    return rearrange(image, "h w c -> c h w")


# 이미지에 텍스트 라벨을 추가하는 함수.
def add_label(
    image: Float[Tensor, "3 width height"],
    label: str,
    font: Path = Path("data/Inter-Regular.otf"),  # 라벨에 사용할 폰트 경로.
    font_size: int = 24,
) -> Float[Tensor, "3 width_with_label height_with_label"]:

    return vcat(  # vertical concatenate
        draw_label(
            label, font, font_size, image.device
        ),  # 라벨 문자열을 이미지로 바꿈.
        image,
        align="left",
        gap=4,
    )
