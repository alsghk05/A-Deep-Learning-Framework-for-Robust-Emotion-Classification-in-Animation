# models_emotion.py

import torch.nn as nn
from torchvision import models

from config import NUM_CLASSES, MODEL_NAME


def create_resnet50(num_classes: int, pretrained: bool = True):
    """
    torchvision resnet50를 가져와 마지막 fc만 num_classes에 맞게 교체
    """
    # torchvision 버전에 따라 weights 인자가 없을 수도 있어서 둘 다 지원
    try:
        backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        )
    except AttributeError:
        # 구버전 torchvision (0.13 이전 등)
        backbone = models.resnet50(pretrained=pretrained)

    in_features = backbone.fc.in_features
    backbone.fc = nn.Linear(in_features, num_classes)
    return backbone


def create_vgg16(num_classes: int, pretrained: bool = True):
    """
    torchvision vgg16을 가져와 classifier 마지막 fc만 num_classes에 맞게 교체
    """
    try:
        backbone = models.vgg16(
            weights=models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        )
    except AttributeError:
        backbone = models.vgg16(pretrained=pretrained)

    in_features = backbone.classifier[-1].in_features
    backbone.classifier[-1] = nn.Linear(in_features, num_classes)
    return backbone


# 🔧 나중에 CBAM / SE 등 커스텀 모델은 여기 추가하면 됨.
# 예시 형태:
#
# def create_resnet50_cbam(num_classes: int, pretrained: bool = True):
#     backbone = create_resnet50(num_classes, pretrained)
#     # backbone.layer1, layer2, layer3, layer4 등에 CBAM 붙이는 코드 추가
#     return backbone


def get_model(model_name=None, num_classes=None, pretrained=True):
    if num_classes is None:
        num_classes = NUM_CLASSES
    if model_name is None:
        model_name = MODEL_NAME

    if model_name == "resnet50":
        return create_resnet50(num_classes, pretrained)

    elif model_name == "vgg16":
        return create_vgg16(num_classes, pretrained)

    # 🔥 CBAM 모델 추가
    elif model_name == "resnet50_cbam":
        return ResNet50_CBAM(num_classes, pretrained)

    elif model_name == "vgg16_cbam":
        return VGG16_CBAM(num_classes, pretrained)

    else:
        raise ValueError(f"지원하지 않는 모델 이름: {model_name}")


import torch
import torch.nn as nn
import torch.nn.functional as F

#############################
# CBAM 모듈 정의
#############################

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_planes, in_planes // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_planes // reduction, in_planes)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.mlp(F.adaptive_avg_pool2d(x, 1).view(b, c))
        max_out = self.mlp(F.adaptive_max_pool2d(x, 1).view(b, c))
        out = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_concat = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv(x_concat))


class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channel_att = ChannelAttention(channels)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x

# models_emotion.py (중간에 추가)

from torchvision import models

class ResNet50_CBAM(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()

        # ResNet50 backbone 가져오기
        try:
            backbone = models.resnet50(
                weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            )
        except:
            backbone = models.resnet50(pretrained=pretrained)

        # Stem 부분 그대로
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        # ResNet block 들
        self.layer1 = backbone.layer1  # out: 256
        self.layer2 = backbone.layer2  # out: 512
        self.layer3 = backbone.layer3  # out: 1024
        self.layer4 = backbone.layer4  # out: 2048

        # CBAM 삽입
        self.cbam1 = CBAM(256)
        self.cbam2 = CBAM(512)
        self.cbam3 = CBAM(1024)
        self.cbam4 = CBAM(2048)

        # pooling & fc
        self.avgpool = backbone.avgpool
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)

        x = self.layer1(x); x = self.cbam1(x)
        x = self.layer2(x); x = self.cbam2(x)
        x = self.layer3(x); x = self.cbam3(x)
        x = self.layer4(x); x = self.cbam4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class VGG16_CBAM(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()

        try:
            backbone = models.vgg16(
                weights=models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
            )
        except:
            backbone = models.vgg16(pretrained=pretrained)

        self.features = backbone.features
        self.avgpool = backbone.avgpool

        # VGG16 block output channels
        # block1 = 64, block2 = 128, block3 = 256, block4 = 512, block5 = 512
        self.cbam3 = CBAM(256)
        self.cbam4 = CBAM(512)
        self.cbam5 = CBAM(512)

        # classifier
        in_features = backbone.classifier[-1].in_features
        backbone.classifier[-1] = nn.Linear(in_features, num_classes)
        self.classifier = backbone.classifier

    def forward(self, x):
        # features는 sequential이어서 index로 CBAM 삽입
        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx == 16:   # block3 끝
                x = self.cbam3(x)
            if idx == 23:   # block4 끝
                x = self.cbam4(x)
            if idx == 30:   # block5 끝
                x = self.cbam5(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
