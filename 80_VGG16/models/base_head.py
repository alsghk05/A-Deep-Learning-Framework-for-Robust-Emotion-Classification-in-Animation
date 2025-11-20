import torch
import torch.nn as nn

# ===========================================================
# Class 1. VGG16 Head
# ===========================================================
class VGG16Head(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
