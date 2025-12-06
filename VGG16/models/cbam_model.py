import torch.nn as nn
from .cbam_backbone import VGG16CBAMBackbone
from .base_head import VGG16Head

# ===========================================================
# VGG16 + CBAM model
# Includes: VGG16CBAMBackbone + VGG16Head
# ===========================================================
class VGG16_CBAM(nn.Module):
    def __init__(
            self,
            num_classes=4,
            use_pretrained=True,
            use_cbam_block3=False,
            use_cbam_block4=True,
            use_cbam_block5=True
    ):
        super().__init__()

        # --- Backbone with CBAM ---
        self.backbone = VGG16CBAMBackbone(
            use_pretrained=use_pretrained,
            use_cbam_block3=use_cbam_block3,
            use_cbam_block4=use_cbam_block4,
            use_cbam_block5=use_cbam_block5
        )

        # --- Head (same classifier as baseline) ---
        self.head = VGG16Head(num_classes=num_classes)

    def forward(self, x):
        x = self.backbone(x)   # Feature map refined by CBAM
        x = self.head(x)       # Final class logits
        return x