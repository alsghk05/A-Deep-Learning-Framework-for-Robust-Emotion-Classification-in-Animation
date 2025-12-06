# ================================================================
# test_Larva.py — Out-of-Domain Test
# using Tom&Jerry-trained BaseModel
# ================================================================
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# (1) Config, Dataset, Metrics import
# ---------------------------------------------------------------
from configs.config_Larva import *   # Larva test directory 설정 필요
from utils.emotion_dataset2 import EmotionDataset, get_transforms
from utils.metrics_emotion import (
    gather_outputs,
    compute_metrics,
    print_metrics,
    save_confusion_matrix,
    save_metrics_to_file
)

# ---------------------------------------------------------------
# (2) backbone+head 통합 모델 가져오기
# ---------------------------------------------------------------
from VGG16.models.base_model import VGG16
from VGG16.models.cbam_model import VGG16_CBAM
from ResNet50.models.base_model import ResNet50

# ================================================================
# main()
# ================================================================
def main():

    print("\n=== Loading Larva Test Dataset ===")
    _, val_transform = get_transforms()

    test_dataset = EmotionDataset(TEST_DIR, EMOTIONS, transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(f"Test samples: {len(test_dataset)}")

    print("\n=== Loading Tom&Jerry-Trained Model ===")
    model = VGG16_CBAM(num_classes=len(EMOTIONS), use_pretrained=False).to(DEVICE)

    ckpt_path = CKPT_PATH   # config_Larva에서 지정
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()

    print("\n=== Running OOD Test on Larva Dataset ===")
    y_true, y_pred, y_prob = gather_outputs(model, test_loader, DEVICE)

    metrics = compute_metrics(y_true, y_pred, y_prob)
    print_metrics(metrics)

    # 결과 저장 폴더
    # out_dir = os.path.join(ROOT_DIR, "experiments_Larva_BaseModel")
    out_dir = os.path.join(ROOT_DIR, "experiments_Larva_CBAMModel")
    os.makedirs(out_dir, exist_ok=True)

    save_metrics_to_file(metrics, out_dir)
    save_confusion_matrix(metrics["confusion_matrix"],
                          EMOTIONS,
                          os.path.join(out_dir, "confusion_matrix_larva.png"))

    print(f"\n[Done] OOD Test Results Saved to: {out_dir}")


if __name__ == "__main__":
    main()
