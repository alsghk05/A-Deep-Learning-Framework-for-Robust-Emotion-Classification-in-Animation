import torch.nn as nn
from base_backbone import VGG16Backbone
from base_head import VGG16Head


# ===========================================================
# Class 1. Full VGG16 Model
# ===========================================================
class VGG16(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.backbone = VGG16Backbone()
        self.head = VGG16Head(num_classes=num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x

