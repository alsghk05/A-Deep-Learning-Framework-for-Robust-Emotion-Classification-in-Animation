# ================================================================
# config_Larva.py — out-of-domain Test for Larva dataset
# ================================================================
import os
import torch

# ------------------------------------------------
# 1) Base Path
# ------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ------------------------------------------------
# 2) Dataset Path — Larva Full Dataset
# ------------------------------------------------
DATASET_DIR = os.path.join(ROOT_DIR, "dataset_Larva")
TEST_DIR = DATASET_DIR  # 전체가 test 용도

# ------------------------------------------------
# 3) Emotion Classes
# ------------------------------------------------
EMOTIONS = ["Anger", "Happiness", "Sadness", "Surprise"]
NUM_CLASSES = len(EMOTIONS)

# ------------------------------------------------
# 4) Device
# ------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------
# 5) BaseModel weight Path (Only TomJerry Train)
# ------------------------------------------------
# CKPT_PATH = os.path.join(
#     ROOT_DIR,
#     "experiments_TomJerry_BaseModel",
#     "1",
#     "TomJerry_full_finetuning_BaseModel.pth"
# )

CKPT_PATH = os.path.join(
    ROOT_DIR,
    "experiments_TomJerry_CBAMModel",
    "1",
    "TomJerry_full_finetuning_CBAMModel.pth"
)


