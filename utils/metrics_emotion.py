import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

from configs.config_TomJerry import EMOTIONS, NUM_CLASSES


# ================================================================
# 1) 모델 출력 수집: y_true, y_pred, y_prob
# ================================================================
def gather_outputs(model, loader, device):
    """
    전체 loader에 대해 y_true, y_pred, y_prob를 수집
    - y_true: (N,)      → 실제 정답 라벨
    - y_pred: (N,)      → 모델이 예측한 클래스 index
    - y_prob: (N, C)    → softmax 확률 (각 클래스별 confidence)
    """

    ## --- 모델을 평가 모드로 변경 ---
    model.eval()

    ## --- 전체 결과를 저장할 리스트 초기화 ---
    all_labels = [] # 모든 정답 라벨 모음
    all_preds = []  # 모든 예측 라벨 모음
    all_probs = []  # 모든 softmax 확률 모음

    with torch.no_grad():
        for images, labels in loader:

            ## --- 입력 이미지와 라벨을 GPU/CPU device로 이동 ---
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)           # (B, C)   → 모델 raw output
            probs = F.softmax(logits, dim=1) # (B, C)   → softmax 확률 계산
            _, preds = torch.max(logits, 1)  # (B,)     → 가장 큰 확률의 클래스 index

            ## --- numpy 배열로 변환하여 리스트에 저장 ---
            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    ## --- batch별 결과를 하나의 큰 배열로 결합 ---
    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)

    return y_true, y_pred, y_prob


# ================================================================
# 2) 메트릭 계산 함수
# ================================================================
def compute_metrics(y_true, y_pred, y_prob):
    """모든 계산을 하나의 dict로 반환"""

    acc = accuracy_score(y_true, y_pred)

    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro"
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted"
    )

    try:
        auc = roc_auc_score(y_true, y_prob, multi_class="ovr")
    except:
        auc = None

    # 클래스별 지표
    prec_cls, rec_cls, f1_cls, support_cls = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=np.arange(NUM_CLASSES)
    )

    per_class = pd.DataFrame({
        "Class": EMOTIONS,
        "Precision": prec_cls,
        "Recall": rec_cls,
        "F1": f1_cls,
        "Support": support_cls,
    })

    return {
        "accuracy": acc,
        "macro_precision": prec_macro,
        "macro_recall": rec_macro,
        "macro_f1": f1_macro,
        "weighted_f1": f1_weighted,
        "auc": auc,
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


# ================================================================
# 3) 메트릭 출력 힘수
# ================================================================
def print_metrics(metrics):

    print("\n===== Overall Metrics =====")
    print(f"Accuracy        : {metrics['accuracy']:.4f}")
    print(f"Macro Precision : {metrics['macro_precision']:.4f}")
    print(f"Macro Recall    : {metrics['macro_recall']:.4f}")
    print(f"Macro F1        : {metrics['macro_f1']:.4f}")
    print(f"Weighted F1     : {metrics['weighted_f1']:.4f}")

    if metrics["auc"] is not None:
        print(f"ROC-AUC (OvR)   : {metrics['auc']:.4f}")
    else:
        print("ROC-AUC (OvR)   : Not Available")

    print("\n===== Per-Class Metrics =====")
    print(metrics["per_class"].to_string(index=False))

    print("\n===== Confusion Matrix =====")
    print(metrics["confusion_matrix"])


# ================================================================
# 4) Confusion Matrix 저장 (Heatmap)
# ================================================================
def save_confusion_matrix(cm, class_names, save_path):

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Test)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ================================================================
# 5) 메트릭 파일 저장
# ================================================================
def save_metrics_to_file(metrics, save_dir):

    ## --- 전체 지표 저장 ---
    summary_path = f"{save_dir}/metrics_summary.txt"
    with open(summary_path, "w") as f:
        for key in ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "auc"]:
            f.write(f"{key}: {metrics[key]:.4f}\n")

    ## --- Per-class 저장 ---
    metrics["per_class"].to_csv(f"{save_dir}/metrics_per_class.csv", index=False)

    print(f">> Metrics saved to {save_dir}")

