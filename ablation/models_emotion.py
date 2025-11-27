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


def get_model(model_name: str = None, num_classes: int = None, pretrained: bool = True):
    """
    model_name: "resnet50", "vgg16" (또는 나중에 "resnet50_cbam" 등 확장)
    """
    if num_classes is None:
        num_classes = NUM_CLASSES
    if model_name is None:
        model_name = MODEL_NAME

    if model_name == "resnet50":
        return create_resnet50(num_classes, pretrained=pretrained)
    elif model_name == "vgg16":
        return create_vgg16(num_classes, pretrained=pretrained)
    # elif model_name == "resnet50_cbam":
    #     return create_resnet50_cbam(num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"지원하지 않는 모델 이름입니다: {model_name}")
