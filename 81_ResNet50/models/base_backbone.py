import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

# ===========================================================
# Helper Function: Conv (for ResNet50) == Used inside ResNet Bottleneck
# ===========================================================
def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels, out_channels,
        kernel_size=1, stride=stride, bias=False
    )


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels, out_channels,
        kernel_size=3, stride=stride, padding=1, bias=False
    )


# ===========================================================
# Bottleneck Block (for ResNet50) == ★ Residual Block ★
# ResNet50은 모든 block이 이 Bottleneck 구조로 구성됨.
# ===========================================================
class Bottleneck(nn.Module):
    """
    Bottleneck block:
      conv1 (1x1): channel compression
      conv2 (3x3): spatial conv
      conv3 (1x1): channel expansion (expansion=4)

      expansion = 4 → mid_channels * 4 == block의 출력 채널 수
    """
    expansion = 4

    # --------------------------------------
    # def 1: conv1, conv2, conv3, ReLU
    # --------------------------------------
    def __init__(self, in_channels, mid_channels, stride=1, downsample=None):
        super().__init__()

        ## --- 1x1 conv: channel compression ---
        self.conv1 = conv1x1(in_channels, mid_channels)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        ## --- 3x3 conv: spatial feature extraction ---
        self.conv2 = conv3x3(mid_channels, mid_channels, stride=stride)
        self.bn2 = nn.BatchNorm2d(mid_channels)

        ## --- 1x1 conv: channel expansion (mid_channels → mid_channels*4) ---
        self.conv3 = conv1x1(mid_channels, mid_channels * self.expansion)
        self.bn3 = nn.BatchNorm2d(mid_channels * self.expansion)

        ## --- shortcut path (for downsample case: stride=2 or channel mismatch) ---
        self.downsample = downsample

        ## --- ReLU ---
        self.relu = nn.ReLU(inplace=True)

    # --------------------------------------
    # def 2: Residual Forward
    # --------------------------------------
    def forward(self, x):
        identity = x    # original input

        ## --- main path ---
        out = self.relu(self.bn1(self.conv1(x)))     # 1x1 compression
        out = self.relu(self.bn2(self.conv2(out)))   # 3x3 spatial conv
        out = self.bn3(self.conv3(out))              # 1x1 expansion

        ## --- shortcut path ---
        if self.downsample is not None:
            identity = self.downsample(x)

        ## --- merge main + shortcut ---
        out += identity
        out = self.relu(out)

        return out


# ===================================================================
# ResNet50 Backbone (Block-based)
# ===================================================================
class ResNet50Backbone(nn.Module):

    # --------------------------------------
    # def 1: Stem, Residual Layer, AdaptiveAvgPool
    # --------------------------------------
    def __init__(self, use_pretrained=True):
        super().__init__()

        ## --- Stem: Convert the input image into initial features ---
        self.conv1 = nn.Conv2d(
            3, 64,
            kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        ## --- Residual Stages --
        self.layer1 = self._make_layer(64,  64, blocks=3, stride=1)     # Bottleneck × 3 (64 → 256)
        self.layer2 = self._make_layer(256, 128, blocks=4, stride=2)    # Bottleneck × 4 (256 → 512)
        self.layer3 = self._make_layer(512, 256, blocks=6, stride=2)    # Bottleneck × 6 (512 → 1024)
        self.layer4 = self._make_layer(1024, 512, blocks=3, stride=2)   # Bottleneck × 3 (1024 → 2048)

        ## --- AdaptiveAvgPool ---
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        ## --- Load pre-trained weight ---
        if use_pretrained:
            self._load_pretrained_resnet50()

    # --------------------------------------
    # def 2: Layer Builder ← consists of several Bottlenecks
    # --------------------------------------
    def _make_layer(self, in_channels, mid_channels, blocks, stride):
        """
        하나의 Stage(layer1~4)를 생성하는 함수.
        첫 블록은 stride/downsample 적용,
        나머지는 stride=1로 반복 구성.
        """
        downsample = None
        out_channels = mid_channels * Bottleneck.expansion

        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                conv1x1(in_channels, out_channels, stride),
                nn.BatchNorm2d(out_channels)
            )

        layers = [Bottleneck(in_channels, mid_channels, stride, downsample)]

        for _ in range(1, blocks):
            layers.append(Bottleneck(out_channels, mid_channels))

        return nn.Sequential(*layers)

    # --------------------------------------
    # def 3: Pretrained weight loader
    # --------------------------------------
    def _load_pretrained_resnet50(self):
        print("[Load]  Loading ImageNet pretrained ResNet50 weights...")

        # ---- 공식 pretrained ResNet50 생성 ----
        official = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        # ---- fc 제외하고 backbone만 로드할 것이므로 strict=False ----
        self.load_state_dict(official.state_dict(), strict=False)

        print("[Success] ResNet50 backbone weights loaded.\n")

    # --------------------------------------
    # def 4: Forward
    # --------------------------------------
    def forward(self, x):

        ## --- Stem: Conv1 + BN1 + ReLU + Maxpool ---
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        ## --- Residual Stages ---
        x = self.layer1(x)  # output Channel: 256
        x = self.layer2(x)  # output Channel: 512
        x = self.layer3(x)  # output Channel: 1024
        x = self.layer4(x)  # output Channel: 2048

        ## --- AdaptiveAvgPool ---
        x = self.avgpool(x)

        return x