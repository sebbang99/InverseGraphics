import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

import torch


class PositionalEncoding(nn.Module):
    def __init__(self, num_octaves: int):
        super().__init__()
        self.num_octaves = num_octaves
        # raise NotImplementedError("This is your homework.")

    def forward(
        self,
        samples: Float[Tensor, "*batch dim"],
    ) -> Float[Tensor, "*batch embedded_dim"]:
        """Separately encode each channel using a positional encoding. The lowest
        frequency should be 2 * torch.pi, and each frequency thereafter should be
        double the previous frequency. For each frequency, you should encode the input
        signal using both sine and cosine.
        """

        # print(f"{samples.shape=}")  # shape [512, 2]

        pos_encoding = []
        for i in range(self.num_octaves):
            frequency = 2**i

            pos_encoding.append(torch.sin(samples * frequency * 2 * torch.pi))
            pos_encoding.append(torch.cos(samples * frequency * 2 * torch.pi))

        pos_encoding = torch.cat(pos_encoding, dim=-1)  # 각 tensor를 결합

        return pos_encoding
        raise NotImplementedError("This is your homework.")

    def d_out(self, dimensionality: int):
        return dimensionality * self.num_octaves * 2
        raise NotImplementedError("This is your homework.")
