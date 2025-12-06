# ================================================================
# config_TomJerry.py — (Base Model) Tom & Jerry Full Fine-Tuning (Base Model)
# (Dataset: 7:2:1 character-preserved split)
# ================================================================

import os
import torch

# ------------------------------------------------
# 1) Base Path
# ------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ------------------------------------------------
# 2) Dataset Path — Tom & Jerry (7:2:1 split)
# ------------------------------------------------
DATASET_DIR = os.path.join(ROOT_DIR, "dataset_TomJerry_split")

TRAIN_DIR = os.path.join(DATASET_DIR, "train")   # 70%
VAL_DIR   = os.path.join(DATASET_DIR, "val")     # 20%
TEST_DIR = os.path.join(DATASET_DIR, "test")     # 10%

# ------------------------------------------------
# 3) Emotion Classes
# ------------------------------------------------
EMOTIONS = ["Anger", "Happiness", "Sadness", "Surprise"]
NUM_CLASSES = len(EMOTIONS)

# ------------------------------------------------
# 4) Training Hyperparameters (Paper-based)
# ------------------------------------------------
LR = 1e-4                 # 논문: 0.0003
BATCH_SIZE = 32           # 논문: 32
NUM_EPOCHS = 60           # 논문: 60
NUM_WORKERS = 0

# ------------------------------------------------
# 5) Device
# ------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------
# 6) Fine-Tuning Options
# ------------------------------------------------
USE_PRETRAINED = True         # ImageNet weights 사용
FREEZE_BACKBONE = False       # Full fine-tuning (Conv1~Conv5 모두 학습)

# ------------------------------------------------
# 7) Loss Function
# ------------------------------------------------
LOSS_NAME = "ce"              # CE가 논문 baseline과 동일
LABEL_SMOOTHING = 0.0         # TomJerry 데이터 충분 → smoothing 불필요
FOCAL_GAMMA = 2.0

# ------------------------------------------------
# 8) Model Selection
# ------------------------------------------------
# MODEL_NAME = "vgg16"
MODEL_NAME = "vgg16_cbam"

# ------------------------------------------------
# 9) Experiment Directory
# ------------------------------------------------
# EXPERIMENT_DIR = os.path.join(ROOT_DIR, "experiments_TomJerry_BaseModel")
EXPERIMENT_DIR = os.path.join(ROOT_DIR, "experiments_TomJerry_CBAMModel")
os.makedirs(EXPERIMENT_DIR, exist_ok=True)

# ------------------------------------------------
# 10) Checkpoint Naming
# ------------------------------------------------
# CKPT_NAME = "TomJerry_full_finetuning_BaseModel"
CKPT_NAME = "TomJerry_full_finetuning_CBAMModel"
