import torch.nn as nn
from .base_backbone import VGG16Backbone
from .base_head import VGG16Head

# ===========================================================
# Full VGG16 model for emotion classification.
# Includes: VGG16Backbone, VGG16Head
# ===========================================================
class VGG16(nn.Module):
    def __init__(self, num_classes=4, use_pretrained=True):
        super().__init__()

        ## --- Backbone (Conv layers + pooling) ---
        self.backbone = VGG16Backbone(use_pretrained=use_pretrained)

        ## --- Head (Classifier: FC layers) ---
        self.head = VGG16Head(num_classes=num_classes)

    def forward(self, x):
        x = self.backbone(x)    # Feature map
        x = self.head(x)        # Class logits
        return x