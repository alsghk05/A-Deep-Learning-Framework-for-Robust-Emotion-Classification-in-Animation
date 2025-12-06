import torch
import torch.nn as nn
from torchvision.models import vgg16

from ..modules.cbam import CBAM # CBAM module

# ===========================================================
# Helper Function: Conv + ReLU (for VGG16)
# ===========================================================
def conv3x3(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )

# ===========================================================
# VGG16 Backbone with CBAM inserted (Block-based)
# (block4, block5 default, but block3 optional)
# ===========================================================
class VGG16CBAMBackbone(nn.Module):

    def __init__(self,
                 use_pretrained=True,
                 use_cbam_block3=False,    # 옵션: block3에 CBAM 삽입 여부
                 use_cbam_block4=True,
                 use_cbam_block5=True):
        super().__init__()

        ## --- Block 1: 64, 64 ---
        self.block1 = nn.Sequential(
            conv3x3(3, 64),
            conv3x3(64, 64),
            nn.MaxPool2d(2, 2)
        )

        ## --- Block 2: 128, 128 ---
        self.block2 = nn.Sequential(
            conv3x3(64, 128),
            conv3x3(128, 128),
            nn.MaxPool2d(2, 2)
        )

        ## --- Block 3: 256, 256, 256 ---
        block3_layers = [
            conv3x3(128, 256),
            conv3x3(256, 256),
            conv3x3(256, 256),
        ]
        if use_cbam_block3:
            block3_layers.append(CBAM(256))
        block3_layers.append(nn.MaxPool2d(2, 2))
        self.block3 = nn.Sequential(*block3_layers)

        ## --- Block 4: 512, 512, 512 ---
        block4_layers = [
            conv3x3(256, 512),
            conv3x3(512, 512),
            conv3x3(512, 512),
        ]
        if use_cbam_block4:
            block4_layers.append(CBAM(512))
        block4_layers.append(nn.MaxPool2d(2, 2))
        self.block4 = nn.Sequential(*block4_layers)

        ## --- Block 5: 512, 512, 512 ---
        block5_layers = [
            conv3x3(512, 512),
            conv3x3(512, 512),
            conv3x3(512, 512),
        ]
        if use_cbam_block5:
            block5_layers.append(CBAM(512))
        block5_layers.append(nn.MaxPool2d(2, 2))
        self.block5 = nn.Sequential(*block5_layers)

        ## --- AdaptiveAvgPool ---
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        ## --- Load pre-trained weight ---
        if use_pretrained:
            self._load_pretrained_vgg()

    # --------------------------------------
    # def 2: Pretrained weight loader
    # --------------------------------------
    def _load_pretrained_vgg(self):
        print("[Load] Loading ImageNet pretrained VGG16 weights...\n")

        official = vgg16(weights="IMAGENET1K_V1")
        official_features = official.features

        ## --- Flatten only Conv Layer of our Backbone Style ---
        own_layers = []
        for block in [self.block1, self.block2, self.block3, self.block4, self.block5]:
            for layer in block:

                ## case 1: con3x3
                if isinstance(layer, nn.Sequential):
                    for sub in layer:
                        if isinstance(sub, nn.Conv2d):
                            own_layers.append(sub)

                ## case 2: independent Conv Layer
                elif isinstance(layer, nn.Conv2d):
                    own_layers.append(layer)

        ## --- Extract only Conv Layers from official VGG ---
        off_layers = [l for l in official_features if isinstance(l, nn.Conv2d)]

        ## --- Mapping pretrained weights to extracted Conv layers---
        for own, off in zip(own_layers, off_layers):
            own.weight.data = off.weight.data.clone()
            if off.bias is not None:
                own.bias.data = off.bias.data.clone()

        print("[Success] Pretrained Conv weights loaded successfully.\n")

    # --------------------------------------
    # def 3: Forward
    # --------------------------------------
    def forward(self, x):

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        x = self.avgpool(x)
        return x