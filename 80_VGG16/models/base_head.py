import torch
import torch.nn as nn

# ===========================================================
# VGG16 Head (Emotion Classification)
# ===========================================================
class VGG16Head(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        ## --- classifier for 4 Emotion ---
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),

            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),

            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
