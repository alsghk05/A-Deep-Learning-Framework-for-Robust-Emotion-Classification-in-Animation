# metrics_emotion.py

import numpy as np
import torch
import torch.nn.functional as F

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

from config import EMOTIONS, NUM_CLASSES


def gather_outputs(model, loader, device):
    """
    전체 loader에 대해 y_true, y_pred, y_prob를 수집
    - y_true: (N,)
    - y_pred: (N,)
    - y_prob: (N, C)  # softmax 확률
    """
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)           # (B, C)
            probs = F.softmax(logits, dim=1) # (B, C)
            _, preds = torch.max(logits, 1)  # (B,)

            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)

    return y_true, y_pred, y_prob


def print_metrics(y_true, y_pred, y_prob, class_names=None):
    """
    전체 성능 + 클래스별 성능 출력
    """
    if class_names is None:
        class_names = EMOTIONS

    # === Overall metrics ===
    acc = accuracy_score(y_true, y_pred)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro"
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted"
    )

    # multi-class AUC(one-vs-rest), 확률 필요
    try:
        auc_ovr = roc_auc_score(y_true, y_prob, multi_class="ovr")
    except Exception:
        auc_ovr = None

    print("\n===== Overall Metrics =====")
    print(f"Accuracy        : {acc:.4f}")
    print(f"Macro Precision : {prec_macro:.4f}")
    print(f"Macro Recall    : {rec_macro:.4f}")
    print(f"Macro F1        : {f1_macro:.4f}")
    print(f"Weighted F1     : {f1_weighted:.4f}")
    if auc_ovr is not None:
        print(f"ROC-AUC (OvR)   : {auc_ovr:.4f}")
    else:
        print("ROC-AUC (OvR)   : 계산 불가 (클래스/확률 문제)")

    # === Per-class metrics ===
    print("\n===== Per-class Metrics =====")
    prec_cls, rec_cls, f1_cls, support_cls = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=np.arange(NUM_CLASSES)
    )

    print(f"{'Class':<12} {'P':>6} {'R':>6} {'F1':>6} {'N':>6}")
    for i, name in enumerate(class_names):
        print(f"{name:<12} {prec_cls[i]:6.3f} {rec_cls[i]:6.3f} {f1_cls[i]:6.3f} {support_cls[i]:6d}")

    # sklearn classification report
    print("\n===== Classification Report =====")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=3))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(NUM_CLASSES))
    print("===== Confusion Matrix (rows=true, cols=pred) =====")
    print(cm)
