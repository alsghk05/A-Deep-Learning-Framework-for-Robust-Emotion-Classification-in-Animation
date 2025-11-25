import torch
import torch.nn as nn

# ===========================================================
# ResNet50 Head (Emotion Classification)
# ===========================================================
class ResNet50Head(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        ## --- classifier for 4 Emotion ---
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = torch.flatten(x, 1)  # (B, 2048)
        x = self.classifier(x)
        return x

