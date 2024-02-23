import torch
import torch.nn as nn
import math
from torch.nn import init
import torch.nn.functional as F
from functools import partial
from collections import OrderedDict


class TimeEmbedding(nn.Module):

    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, t):
        emb = self.timembedding(t)
        return emb


class AttentionBlock(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        # self.group_norm = nn.GroupNorm(32, in_ch)
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.initialize()

    def initialize(self):
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)
        init.xavier_uniform_(self.proj.weight, gain=1e-5)

    def forward(self, x):
        B, C, H, W = x.shape
        # h = self.group_norm(x)
        q = self.proj_q(x)
        k = self.proj_k(x)
        v = self.proj_v(x)

        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.view(B, C, H * W)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        assert list(w.shape) == [B, H * W, H * W]
        w = F.softmax(w, dim=-1)

        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)

        return x + h


class ResidualBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, t_dim: int, self_attention: bool = False):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.t_dim = t_dim
        self.self_attention = self_attention

        self.block_1 = nn.Sequential(
            # nn.GroupNorm(32, self.in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                      kernel_size=3, stride=1, padding=1),
        )

        self.temp_projection = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.t_dim, self.out_channels)
        )

        self.block_2 = nn.Sequential(
            # nn.GroupNorm(32, self.out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.out_channels,
                      out_channels=self.out_channels, kernel_size=3, stride=1, padding=1),
        )

        self.shortcut = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                  kernel_size=1) if self.in_channels != self.out_channels else nn.Identity()
        if self.self_attention:
            self.attention = AttentionBlock(self.out_channels)
        else:
            self.attention = nn.Identity()

    def forward(self, input: torch.Tensor, t_pos: torch.Tensor) -> torch.Tensor:

        output1 = self.block_1(input)
        temp_pos = self.temp_projection(t_pos)[:, :, None, None]
        output2 = self.block_2(output1 + temp_pos)
        res = output2 + self.shortcut(input)
        attention = self.attention(res)

        return attention


class DownSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb):
        x = self.main(x)
        return x


class Unet(nn.Module):

    def __init__(self, n_channels: int, t_length: int, n_res_blocks: int, img_chanels: int = 3, ch_mults=[1, 2, 2, 2]):

        super().__init__()

        self.n_channels = n_channels
        self.t_length = t_length
        self.n_res_blocks = n_res_blocks
        self.t_dim = self.n_channels * 4
        self.img_channels = img_chanels
        self.ch_mults = ch_mults

        self.input_layer = nn.Conv2d(
            in_channels=self.img_channels, out_channels=self.n_channels, kernel_size=3, stride=1, padding=1)

        self.time_embedding = TimeEmbedding(
            self.t_length, self.n_channels, self.t_dim)

        self.down_block = nn.ModuleList()

        channels = [self.n_channels]
        in_channels = self.n_channels
        for i, ch_mult in enumerate(self.ch_mults):
            out_channels = ch_mult * self.n_channels
            for _ in range(n_res_blocks):
                self.down_block.append(ResidualBlock(
                    in_channels, out_channels, self.t_dim, True))
                in_channels = out_channels
                channels.append(in_channels)

            if i != len(self.ch_mults) - 1:
                self.down_block.append(DownSample(in_channels))
                channels.append(in_channels)

        self.middle_block = nn.ModuleList(
            [
                ResidualBlock(in_channels=in_channels, out_channels=in_channels,
                              t_dim=self.t_dim, self_attention=True),
                ResidualBlock(in_channels=in_channels, out_channels=in_channels,
                              t_dim=self.t_dim, self_attention=False),
            ]
        )

        self.up_block = nn.ModuleList()

        for i, ch_mult in reversed(list(enumerate(self.ch_mults))):
            out_channels = ch_mult * self.n_channels
            for _ in range(n_res_blocks + 1):
                self.up_block.append(ResidualBlock(
                    in_channels + channels.pop(), out_channels, self.t_dim, True
                ))
                in_channels = out_channels

            if i != 0:
                self.up_block.append(nn.Upsample(
                    scale_factor=2, mode="nearest"))

        self.output = nn.Sequential(
            # nn.GroupNorm(32, in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, self.img_channels,
                      kernel_size=3, stride=1, padding=1)

        )
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.input_layer.weight)
        init.zeros_(self.input_layer.bias)
        init.xavier_uniform_(self.output[-1].weight, gain=1e-5)
        init.zeros_(self.output[-1].bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:

        temp = self.time_embedding(t)
        h = self.input_layer(x)
        outputs = [h]
        for layer in self.down_block:
            h = layer(h, temp)
            outputs.append(h)

        for layer in self.middle_block:
            h = layer(h, temp)

        for layer in self.up_block:
            if isinstance(layer, ResidualBlock):
                h = torch.cat([h, outputs.pop()], dim=1)
                h = layer(h, temp)
            else:
                h = layer(h)
        h = self.output(h)
        return h
