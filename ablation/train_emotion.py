# train_emotion.py
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt  # ✅ 그래프 그리기

from config import DEVICE, LR, NUM_EPOCHS, MODEL_NAME
from emotion_dataset import get_dataloaders
from models_emotion import get_model
from losses_emotion import get_loss
from metrics_emotion import gather_outputs, print_metrics


def train_one_epoch(epoch, model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    running_correct = 0
    total = 0

    start = time.time()

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)        # (B, num_classes)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        running_correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_correct / total

    print(f"[Train] Epoch {epoch}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.4f}, "
          f"Time={time.time()-start:.1f}s")
    return epoch_loss, epoch_acc


def eval_one_epoch(epoch, model, loader, criterion, device, mode="Val"):
    """
    에포크 중간에는 val loss/acc만 빠르게 본다.
    (정밀한 F1, AUC 등은 학습 끝난 후 best model로 한 번에 계산)
    """
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            running_correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_correct / total

    print(f"[{mode}] Epoch {epoch}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.4f}")
    return epoch_loss, epoch_acc


def make_run_folder(base_dir="./experiments"):
    """
    ./experiments/1, 2, 3 ... 이런 식으로
    아직 없는 번호를 찾아서 새 폴더를 만들어줌.
    """
    os.makedirs(base_dir, exist_ok=True)
    run_id = 1
    while os.path.exists(os.path.join(base_dir, str(run_id))):
        run_id += 1
    run_dir = os.path.join(base_dir, str(run_id))
    os.makedirs(run_dir, exist_ok=True)
    print(f"\n=== 이번 실험 결과는 폴더에 저장됩니다: {run_dir} ===\n")
    return run_dir


def plot_history(history, run_dir):
    """
    history: {"train_loss": [...], "val_loss": [...],
              "train_acc": [...],  "val_acc": [...]}
    각 지표별로 train/val 그래프를 epoch에 따라 저장.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    # 1) Loss 그래프
    plt.figure()
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Val Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "loss_curve.png"))
    plt.close()

    # 2) Accuracy 그래프
    plt.figure()
    plt.plot(epochs, history["train_acc"], label="train_acc")
    plt.plot(epochs, history["val_acc"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Train vs Val Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "accuracy_curve.png"))
    plt.close()


def main():
    # 0) 실험 폴더 생성 (1,2,3...)
    run_dir = make_run_folder(base_dir="./experiments")

    # 1) 데이터
    train_loader, val_loader, class_to_idx = get_dataloaders()
    print("class_to_idx:", class_to_idx)

    # 2) 모델
    model = get_model(MODEL_NAME).to(DEVICE)
    print("사용 모델:", MODEL_NAME)

    # 3) loss & optimizer
    criterion = get_loss()
    print("사용 Loss:", type(criterion).__name__)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0
    save_path = os.path.join(run_dir, f"{MODEL_NAME}_best.pth")

    history = {"train_loss": [], "train_acc": [],
               "val_loss": [], "val_acc": []}

    # 4) 학습 루프
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            epoch, model, train_loader, optimizer, criterion, DEVICE
        )
        val_loss, val_acc = eval_one_epoch(
            epoch, model, val_loader, criterion, DEVICE, mode="Val"
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # best model 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "class_to_idx": class_to_idx,
                    "history": history,
                },
                save_path,
            )
            print(f"👉 Best model 갱신: {save_path} (val_acc={val_acc:.4f})")

    print("최고 Val 정확도:", best_val_acc)

    # 5) 🔥 history 그래프 저장 (loss/acc)
    plot_history(history, run_dir)

    # 6) Best model 로드 후, 정밀 성능 평가 (F1, Precision, Recall, AUC 등)
    print("\n=== Best model 로드 후 정밀 평가 ===")
    checkpoint = torch.load(save_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    y_true, y_pred, y_prob = gather_outputs(model, val_loader, DEVICE)
    # 콘솔에 지표 출력
    print_metrics(y_true, y_pred, y_prob)

    # (선택) 나중에 필요하면 text 파일로도 저장 가능:
    # with open(os.path.join(run_dir, "metrics.txt"), "w") as f:
    #     ...  # classification_report를 문자열로 받아서 저장하면 됨


if __name__ == "__main__":
    main()
