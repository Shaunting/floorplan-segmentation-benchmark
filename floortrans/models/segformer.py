import torch
import torch.nn as nn
import torch.nn.functional as F


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_ch, out_ch, patch_size, stride):
        super().__init__()

        self.proj = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2
        )

        self.norm = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class EfficientSelfAttention(nn.Module):
    def __init__(self, dim, heads=4, reduction=8):
        super().__init__()

        self.heads = heads
        self.scale = (dim // heads) ** -0.5

        self.q = nn.Conv2d(dim, dim, 1)
        self.kv = nn.Conv2d(dim, dim * 2, 1)

        self.reduction = reduction

        if reduction > 1:
            self.sr = nn.Conv2d(
                dim,
                dim,
                kernel_size=reduction,
                stride=reduction
            )
            self.norm = nn.BatchNorm2d(dim)

        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape

        q = self.q(x)
        q = q.reshape(B, self.heads, C // self.heads, H * W)
        q = q.permute(0, 1, 3, 2)

        if self.reduction > 1:
            kv_in = self.sr(x)
            kv_in = self.norm(kv_in)
        else:
            kv_in = x

        _, _, Hr, Wr = kv_in.shape

        kv = self.kv(kv_in)
        kv = kv.reshape(
            B,
            2,
            self.heads,
            C // self.heads,
            Hr * Wr
        )

        k, v = kv[:, 0], kv[:, 1]

        attn = torch.softmax(
            (q @ k) * self.scale,
            dim=-1
        )

        out = attn @ v.transpose(-2, -1)

        out = out.permute(0, 1, 3, 2)
        out = out.reshape(B, C, H, W)

        out = self.proj(out)

        return out


class MixFFN(nn.Module):
    def __init__(self, dim, expansion=4):
        super().__init__()

        hidden = dim * expansion

        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),
            nn.GELU(),
            nn.Conv2d(hidden, dim, 1)
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=4, reduction=8):
        super().__init__()

        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = EfficientSelfAttention(
            dim,
            heads=heads,
            reduction=reduction
        )

        self.norm2 = nn.BatchNorm2d(dim)
        self.ffn = MixFFN(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class segFormer(nn.Module):
    def __init__(self, n_classes):
        super().__init__()

        self.stage1 = OverlapPatchEmbed(3, 64, 7, 4)
        self.block1 = TransformerBlock(64, heads=1, reduction=8)

        self.stage2 = OverlapPatchEmbed(64, 128, 3, 2)
        self.block2 = TransformerBlock(128, heads=2, reduction=4)

        self.stage3 = OverlapPatchEmbed(128, 256, 3, 2)
        self.block3 = TransformerBlock(256, heads=4, reduction=2)

        self.stage4 = OverlapPatchEmbed(256, 512, 3, 2)
        self.block4 = TransformerBlock(512, heads=8, reduction=1)

        self.decode = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, n_classes, 1)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = self.stage1(x)
        x = self.block1(x)

        x = self.stage2(x)
        x = self.block2(x)

        x = self.stage3(x)
        x = self.block3(x)

        x = self.stage4(x)
        x = self.block4(x)

        x = self.decode(x)

        x = F.interpolate(
            x,
            scale_factor=32,
            mode='bilinear',
            align_corners=False
        )

        x[:, :21] = self.sigmoid(x[:, :21])

        return x