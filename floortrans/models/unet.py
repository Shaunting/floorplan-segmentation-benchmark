import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Basic Conv Block
# -------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


# -------------------------
# UNet
# -------------------------
class UNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()

        # Encoder
        self.enc1 = DoubleConv(3, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)

        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        # IMPORTANT: must match Furukawa split
        # 21 heatmaps + 12 rooms + 11 icons = 44
        self.out = nn.Conv2d(64, n_classes, kernel_size=1)

        self._init_weights()

    # -------------------------
    # FIX: safe skip connection
    # -------------------------
    def _align_and_concat(self, x, skip):
        _, _, H, W = skip.shape
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        return torch.cat([x, skip], dim=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder + SAFE SKIP CONNECTIONS
        d4 = self.up4(b)
        d4 = self.dec4(self._align_and_concat(d4, e4))

        d3 = self.up3(d4)
        d3 = self.dec3(self._align_and_concat(d3, e3))

        d2 = self.up2(d3)
        d2 = self.dec2(self._align_and_concat(d2, e2))

        d1 = self.up1(d2)
        d1 = self.dec1(self._align_and_concat(d1, e1))

        out = self.out(d1)

        # Match Furukawa behavior: heatmaps sigmoid
        out = out.clone()
        out[:, :21] = torch.sigmoid(out[:, :21])

        return out

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)