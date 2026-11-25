"""Recovered ResUNet-18/34/50 with CRSM-compatible checkpoint keys."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Type

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from StabilityAnalyzer import StabilityAnalyzer


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, 3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class ResNetEncoder(nn.Module):
    def __init__(self, block: Type[nn.Module], layers: List[int], ini: int = 3):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(ini, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.in1 = nn.InstanceNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, 1000)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                fan = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                module.weight.data.normal_(0, math.sqrt(2.0 / fan))
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _make_layer(self, block: Type[nn.Module], planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        expansion = int(getattr(block, "expansion"))
        downsample = None
        if stride != 1 or self.inplanes != planes * expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * expansion),
            )
        modules: List[nn.Module] = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * expansion
        modules.extend(block(self.inplanes, planes) for _ in range(1, blocks))
        return nn.Sequential(*modules)

    def forward(self, x: Tensor):
        skips: List[Tensor] = []
        x = self.relu(self.bn1(self.conv1(x)))
        skips.append(x)
        x = self.maxpool(x)
        x = self.layer1(x); skips.append(x)
        x = self.layer2(x); skips.append(x)
        x = self.layer3(x); skips.append(x)
        x = self.layer4(x); skips.append(x)
        return x, skips


class UnetBlock(nn.Module):
    def __init__(self, up_in: int, x_in: int, n_out: int):
        super().__init__()
        self.x_conv = nn.Conv2d(x_in, n_out // 2, 1)
        self.tr_conv = nn.ConvTranspose2d(up_in, n_out // 2, 2, stride=2)
        self.bn = nn.BatchNorm2d(n_out)

    def forward(self, up_p: Tensor, x_p: Tensor) -> Tensor:
        return self.bn(F.relu(torch.cat([self.tr_conv(up_p), self.x_conv(x_p)], dim=1)))


class ResUNet(nn.Module):
    CHANNELS = {
        "resunet18": ([64, 64, 128, 256, 512], BasicBlock, [2, 2, 2, 2]),
        "resunet34": ([64, 64, 128, 256, 512], BasicBlock, [3, 4, 6, 3]),
        "resunet50": ([64, 256, 512, 1024, 2048], Bottleneck, [3, 4, 6, 3]),
    }

    def __init__(
        self,
        name: str,
        num_classes: int = 2,
        use_crsm: bool = True,
        pretrained: bool = True,
        pretrained_path: str | None = None,
        **crsm_kwargs,
    ):
        super().__init__()
        channels, block, layers = self.CHANNELS[name]
        self.res = ResNetEncoder(block, layers)
        self.num_classes = num_classes
        self.use_crsm = use_crsm
        self.up1 = UnetBlock(channels[4], channels[3], 256)
        self.up2 = UnetBlock(256, channels[2], 256)
        self.up3 = UnetBlock(256, channels[1], 256)
        self.up4 = UnetBlock(256, channels[0], 256)
        self.up5 = nn.ConvTranspose2d(256, 32, 2, stride=2)
        self.bnout = nn.BatchNorm2d(32)
        self.seg_head = nn.Conv2d(32, num_classes, 1)
        if use_crsm:
            self.stabilityAnalyzer = StabilityAnalyzer(input_dim=channels[4], **crsm_kwargs)
        if pretrained:
            if pretrained_path:
                state = torch.load(pretrained_path, map_location="cpu")
            else:
                pretrained_urls = {
                    "resunet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
                    "resunet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
                    "resunet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
                }
                state = torch.hub.load_state_dict_from_url(
                    pretrained_urls[name],
                    model_dir=None, map_location="cpu", check_hash=True,
                )
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            state = {key.replace("module.", "", 1): value for key, value in state.items()}
            self.res.load_state_dict(state, strict=False)

    def forward(self, x: Tensor, compute_aux: bool = True) -> Dict[str, object]:
        bottleneck, skips = self.res(x)
        zero = bottleneck.sum() * 0.0
        aux: Dict[str, object] = {"loss_aux": zero, "loss_pert": zero, "loss_stab": zero}
        if self.use_crsm and compute_aux:
            aux = self.stabilityAnalyzer(bottleneck)
        # Decoder intentionally consumes the untouched bottleneck.
        decoded = F.relu(bottleneck)
        decoded = self.up1(decoded, skips[3])
        decoded = self.up2(decoded, skips[2])
        decoded = self.up3(decoded, skips[1])
        decoded = self.up4(decoded, skips[0])
        decoded = self.up5(decoded)
        logits = self.seg_head(F.relu(self.bnout(decoded)))
        return {
            "logits": logits,
            "loss_aux": aux["loss_aux"],
            "loss_pert": aux["loss_pert"],
            "loss_stab": aux["loss_stab"],
            "loss_seg": None,
            "bottleneck": bottleneck,
        }
