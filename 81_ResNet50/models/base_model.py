import torch.nn as nn
from base_backbone import ResNet50Backbone
from base_head import ResNet50Head

# ===========================================================
# Full ResNet50 model for emotion classification.
# Includes: ResNet50Backbone + ResNet50Head
# ===========================================================
class ResNet50(nn.Module):
    def __init__(self, num_classes=4, use_pretrained=True):
        super().__init__()

        ## --- Backbone (Conv layers + residual blocks + avgpool) ---
        self.backbone = ResNet50Backbone(use_pretrained=use_pretrained)

        ## --- Head (Classifier: FC layers) ---
        self.head = ResNet50Head(num_classes=num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x