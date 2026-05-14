import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


class ASPPConv(nn.Sequential):
    def __init__(self, in_ch, out_ch, dilation):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

class ASPPPooling(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        size = x.shape[-2:]
        return F.interpolate(self.pool(x), size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch=256, dilations=(6, 12, 18)):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)),
            ASPPConv(in_ch, out_ch, dilations[0]),
            ASPPConv(in_ch, out_ch, dilations[1]),
            ASPPConv(in_ch, out_ch, dilations[2]),
            ASPPPooling(in_ch, out_ch),
        ])
        self.project = nn.Sequential(
            nn.Conv2d(5 * out_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
    def forward(self, x):
        return self.project(torch.cat([c(x) for c in self.convs], dim=1))


class DeepLabV3Plus(nn.Module):
    def __init__(self, n_classes=44):
        super().__init__()
        backbone = resnet50(pretrained=False)

        # Dilate layer4: output_stride 32 -> 16
        for m in backbone.layer4.modules():
            if isinstance(m, nn.Conv2d):
                if m.stride == (2, 2):
                    m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (2, 2)
                    m.padding = (2, 2)

        # Manual layer extraction — no IntermediateLayerGetter needed
        self.stem   = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1   # 256 ch, 1/4 res  (low-level features)
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4   # 2048 ch, 1/16 res (high-level features)

        self.low_proj = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.aspp = ASPP(2048, 256)
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, n_classes, 1),
        )
        self._init_weights()

    def forward(self, x):
        input_size = x.shape[-2:]
        x       = self.stem(x)
        low     = self.layer1(x)
        x       = self.layer2(low)
        x       = self.layer3(x)
        x       = self.layer4(x)

        low_feat  = self.low_proj(low)
        high_feat = F.interpolate(self.aspp(x), size=low_feat.shape[-2:], mode='bilinear', align_corners=False)

        out = self.decoder(torch.cat([high_feat, low_feat], dim=1))
        out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=False)
        out = out.clone()
        out[:, :21] = torch.sigmoid(out[:, :21])
        return out

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def deepLab(n_classes=44):
    return DeepLabV3Plus(n_classes=n_classes)