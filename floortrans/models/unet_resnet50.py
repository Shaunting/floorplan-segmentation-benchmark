import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


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


class UNetResNet50(nn.Module):
    def __init__(self, n_classes):
        super().__init__()

        backbone = resnet50(pretrained=True)

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1,
                                  backbone.relu, backbone.maxpool)
        self.enc1 = backbone.layer1   # H/4,  256ch
        self.enc2 = backbone.layer2   # H/8,  512ch
        self.enc3 = backbone.layer3   # H/16, 1024ch
        self.enc4 = backbone.layer4   # H/32, 2048ch  (bottleneck)

        # Decoder
        self.up3  = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(2048, 1024)   # 1024 up + 1024 skip

        self.up2  = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(1024, 512)    # 512 up + 512 skip

        self.up1  = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(512, 256)     # 256 up + 256 skip

        # Furukawa-aligned head: predict at H/4, then 4x upsample to full res
        self.head     = nn.Conv2d(256, n_classes, kernel_size=1)
        self.upsample = nn.ConvTranspose2d(n_classes, n_classes,
                                           kernel_size=4, stride=4)

        self._init_decoder_weights()

    def _align_and_concat(self, x, skip):
        _, _, H, W = skip.shape
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        return torch.cat([x, skip], dim=1)

    def forward(self, x):
        # Encoder
        s  = self.stem(x)    # H/4,  64ch
        e1 = self.enc1(s)    # H/4,  256ch
        e2 = self.enc2(e1)   # H/8,  512ch
        e3 = self.enc3(e2)   # H/16, 1024ch
        e4 = self.enc4(e3)   # H/32, 2048ch

        # Decoder
        d3 = self.up3(e4)
        d3 = self.dec3(self._align_and_concat(d3, e3))

        d2 = self.up2(d3)
        d2 = self.dec2(self._align_and_concat(d2, e2))

        d1 = self.up1(d2)
        d1 = self.dec1(self._align_and_concat(d1, e1))

        # Head
        out = self.upsample(self.head(d1))

        # Match both existing models: heatmap channels through sigmoid
        out = out.clone()
        out[:, :21] = torch.sigmoid(out[:, :21])
        return out

    def _init_decoder_weights(self):
        decoder_modules = [
            self.up3, self.dec3,
            self.up2, self.dec2,
            self.up1, self.dec1,
            self.head, self.upsample,
        ]
        for module in decoder_modules:
            for m in module.modules():
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                           nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
