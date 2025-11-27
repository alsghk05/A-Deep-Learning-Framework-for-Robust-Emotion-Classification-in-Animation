# config.py

import os
import torch

# === 경로 설정 ===
# metadata_down.csv, train/, test/ 가 있는 상위 경로로 바꿔줘
DATA_ROOT = "./preprocessed_final_data_down/Dinocore_preprocessed_down/"

TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VAL_DIR   = os.path.join(DATA_ROOT, "test")

# 감정 클래스
EMOTIONS = ["Anger", "Happiness", "Sadness", "Surprise"]
NUM_CLASSES = len(EMOTIONS)

# 하이퍼파라미터
BATCH_SIZE = 32
NUM_EPOCHS = 10
LR = 1e-4
NUM_WORKERS = 4

# 모델 이름
MODEL_NAME = "resnet50"   # "resnet50", "vgg16", 나중에 "resnet50_cbam" 등 추가 가능

# 🔥 사용할 Loss 종류
#  - "ce"      : 기본 CrossEntropy
#  - "ce_ls"   : Label Smoothing CrossEntropy
#  - "focal"   : Focal Loss
LOSS_NAME = "ce"          # 필요할 때 "ce_ls", "focal" 로만 바꿔주면 됨
LABEL_SMOOTHING = 0.1     # ce_ls일 때 사용
FOCAL_GAMMA = 2.0         # focal일 때 사용

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

