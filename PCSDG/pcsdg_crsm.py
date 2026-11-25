"""PCSDG-style pixel disentanglement + the project's unchanged CRSM losses."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models as tv_models

from StabilityAnalyzer import StabilityAnalyzer


def make_resnet(name: str, pretrained: bool):
    fn = getattr(tv_models, name)
    try:  # torchvision >= 0.13
        weights_enum = getattr(tv_models, name.replace("resnet", "ResNet") + "_Weights", None)
        return fn(weights=weights_enum.DEFAULT if pretrained and weights_enum else None)
    except (TypeError, AttributeError):  # torchvision used on the original server
        return fn(pretrained=pretrained)


class DWM(nn.Module):
    """PCSDG's pixel-level content/style weight-map encoder-decoder."""
    def __init__(self, out_channels=2):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True))
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True))
        self.enc3 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True))
        self.pool = nn.MaxPool2d(2)
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.up3 = nn.ConvTranspose2d(64, out_channels, 2, stride=2)

    def forward(self, x):
        x = self.pool(self.enc1(x))
        x = self.pool(self.enc2(x))
        x = self.pool(self.enc3(x))
        return torch.sigmoid(self.up3(self.up2(self.up1(x))))


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.x_conv = nn.Conv2d(skip_ch, out_ch // 2, 1)
        self.tr_conv = nn.ConvTranspose2d(in_ch, out_ch // 2, 2, stride=2)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x, skip):
        x = self.tr_conv(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.bn(F.relu(torch.cat([x, self.x_conv(skip)], dim=1)))


class PCSDGCRSM(nn.Module):
    """PCSDG DWM -> shared ResNet encoder/UNet decoder, plus loss-only CRSM."""
    def __init__(self, task="odoc", backbone=None, pretrained=True,
                 use_crsm=True,
                 crsm_alpha=.3, sigma=.05, gamma=.1, t_pert=3, t_stab=2,
                 num_perturb=4):
        super().__init__()
        if backbone is None:
            backbone = "resnet34" if task == "odoc" else "resnet50"
        self.task, self.backbone_name, self.use_crsm = task, backbone, use_crsm
        self.num_classes = 2 if task == "odoc" else 1
        rn = make_resnet(backbone, pretrained)
        self.stem_conv, self.stem_bn, self.stem_relu, self.pool = rn.conv1, rn.bn1, rn.relu, rn.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = rn.layer1, rn.layer2, rn.layer3, rn.layer4
        exp = 4 if backbone in ("resnet50", "resnet101", "resnet152") else 1
        channels = [64, 64 * exp, 128 * exp, 256 * exp, 512 * exp]
        self.dwm = DWM()
        self.up1 = UpBlock(channels[4], channels[3], 256)
        self.up2 = UpBlock(256, channels[2], 256)
        self.up3 = UpBlock(256, channels[1], 256)
        self.up4 = UpBlock(256, channels[0], 256)
        self.seg_head = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)
        # Preserve PCSDG's 8x56x56 -> 1024 projector. Adaptive max-pooling only
        # reconciles the paper's 224 input with this project's 512 input size.
        self.projector = nn.Sequential(nn.Conv2d(64, 8, 3, padding=1), nn.BatchNorm2d(8),
                                       nn.ReLU(inplace=True), nn.MaxPool2d(2),
                                       nn.AdaptiveMaxPool2d((56, 56)), nn.Flatten(),
                                       nn.Linear(8 * 56 * 56, 1024))
        self.stabilityAnalyzer = None
        if use_crsm:
            self.stabilityAnalyzer = StabilityAnalyzer(
                input_dim=channels[4], hidden_dim=256, t_pert=t_pert, t_stab=t_stab,
                sigma=sigma, num_perturb=num_perturb, gamma=gamma, alpha=crsm_alpha)

    def encode(self, x):
        stem = self.stem_relu(self.stem_bn(self.stem_conv(x)))
        x = self.pool(stem)
        s1 = self.layer1(x)
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        bottleneck = self.layer4(s3)
        return bottleneck, (stem, s1, s2, s3)

    def forward(self, image, compute_aux=True, return_embeddings=True):
        weights = self.dwm(image)
        content = image * weights[:, 0:1]
        style = image * weights[:, 1:2]
        bottleneck, skips = self.encode(content)
        logits = self.seg_head(self.up4(self.up3(self.up2(self.up1(bottleneck, skips[3]), skips[2]), skips[1]), skips[0]))
        if logits.shape[-2:] != image.shape[-2:]:
            logits = F.interpolate(logits, size=image.shape[-2:], mode="bilinear", align_corners=False)
        aux = {"loss_aux": bottleneck.sum() * 0, "loss_pert": bottleneck.sum() * 0,
               "loss_stab": bottleneck.sum() * 0}
        if self.stabilityAnalyzer is not None and compute_aux:
            aux = self.stabilityAnalyzer(bottleneck)
        # The auxiliary content/style features are the PCSDG contrastive representations.
        result = {"logits": logits, "bottleneck": bottleneck, **aux}
        if return_embeddings:
            # The public PCSDG forward_DisentangleWeightMap returns the first
            # ResNet convolution output, before BN/ReLU.
            f_content = self.stem_conv(content)
            f_style = self.stem_conv(style)
            result["content_embed"] = F.normalize(self.projector(f_content), dim=1)
            result["style_embed"] = F.normalize(self.projector(f_style), dim=1)
        return result

    def get_disentangled_embeddings(self, image):
        weights = self.dwm(image)
        content = image * weights[:, 0:1]
        style = image * weights[:, 1:2]
        f_content = self.stem_conv(content)
        f_style = self.stem_conv(style)
        return (F.normalize(self.projector(f_content), dim=1),
                F.normalize(self.projector(f_style), dim=1))


def pcsdg_pair_losses(embeddings):
    """Symmetric pairwise content invariance and style separation used by PCSDG."""
    content = [item[0] for item in embeddings]
    style = [item[1] for item in embeddings]
    content_loss = content[0].new_zeros(())
    style_loss = content_loss.clone()
    for i in range(len(content)):
        for j in range(i + 1, len(content)):
            content_loss = content_loss + 2 * F.l1_loss(content[i], content[j])
            style_loss = style_loss - 2 * F.l1_loss(style[i], style[j])
    return content_loss, style_loss
