# ================================================================
# train_TomJerry.py — (Base Model) Tom & Jerry Full Fine-Tuning
#
# 본 스크립트 코드의 핵심 목적:
#   1) Tom&Jerry (7:2:1 split) 데이터로 base model(VGG16 or ResNet50) 전체 미세조정(fine-tuning)
#   2) Validation 데이터는 "모델 선택용" (best checkpoint 저장)
#   3) Test 데이터는 "최종 성능 보고용" (논문에 제시될 accuracy/F1/CM 등)
# ================================================================
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# (1) Config, Dataset, Loss, Metrics import
# ---------------------------------------------------------------
from configs.config_TomJerry import *
from utils.emotion_dataset2 import get_dataloaders
from utils.losses_emotion import get_loss
from utils.metrics_emotion import (
    gather_outputs,
    compute_metrics,
    print_metrics,
    save_confusion_matrix,
    save_metrics_to_file,
)

# ---------------------------------------------------------------
# (2) backbone+head 통합 모델 가져오기
# ---------------------------------------------------------------
from VGG16.models.base_model import VGG16
from VGG16.models.cbam_model import VGG16_CBAM
from ResNet50.models.base_model import ResNet50

# ================================================================
# get_model(): 모델 이름만 입력하면 적절한 모델 객체를 반환
# ================================================================
def get_model(model_name: str):

    model_name = model_name.lower()

    if model_name == "vgg16":
        return VGG16(num_classes=NUM_CLASSES, use_pretrained=USE_PRETRAINED)

    elif model_name == "vgg16_cbam":
        return VGG16_CBAM(
            num_classes=NUM_CLASSES,
            use_pretrained=USE_PRETRAINED,
            use_cbam_block3=False,
            use_cbam_block4=True,
            use_cbam_block5=True
        )

    elif model_name == "resnet50":
        return ResNet50(num_classes=NUM_CLASSES, use_pretrained=USE_PRETRAINED)

    else:
        raise ValueError(f"지원하지 않는 MODEL_NAME: {model_name}")


# ================================================================
# 실험 결과 폴더 생성 (1,2,3....)
# ================================================================
# def make_run_folder(base_dir="./experiments_TomJerry_BaseModel"):
def make_run_folder(base_dir="./experiments_TomJerry_CBAMModel"):
    """
    실험 결과 저장 폴더 자동 생성:
    experiments_TomJerry_BaseModel/
        ├── 1/
        ├── 2/
        ├── 3/
        ...
    """

    os.makedirs(base_dir, exist_ok=True)

    run_id = 1
    while os.path.exists(os.path.join(base_dir, str(run_id))):
        run_id += 1

    run_dir = os.path.join(base_dir, str(run_id))
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n=== Tom&Jerry 실험 결과 저장 폴더: {run_dir} ===")
    return run_dir


# ================================================================
# 학습 그래프(Loss & Accuracy) 저장
# ================================================================
def plot_history(history, run_dir):
    epochs = range(1, len(history["train_loss"]) + 1)

    ## --- Loss Curve ---
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

    ## --- Accuracy Curve ---
    plt.figure()
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], label="Val Accuracy")
    plt.plot(epochs, history["test_acc"], label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Train vs Val vs Test Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "accuracy_curve.png"))
    plt.close()


# ================================================================
# Train 1 epoch
# ================================================================
def train_one_epoch(epoch, model, loader, optimizer, criterion, device):

    model.train()
    running_loss = 0.0
    running_correct = 0
    total = 0
    start = time.time()

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        running_correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_correct / total

    print(f"[Train] Epoch {epoch} | Loss={epoch_loss:.4f} | Acc={epoch_acc:.4f} "
          f"| Time={time.time() - start:.1f}s")

    return epoch_loss, epoch_acc


# ================================================================
# eval_one_epoch(): 한 epoch 검증
# ================================================================
def eval_one_epoch(epoch, model, loader, criterion, device, tag="Val"):

    model.eval()
    running_loss = 0.0
    running_correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            running_correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_correct / total

    print(f"[{tag}] Epoch {epoch} | Loss={epoch_loss:.4f} | Acc={epoch_acc:.4f}")

    return epoch_loss, epoch_acc


# ================================================================
# main()
# ================================================================
def main():

    # -----------------------------------------------------------
    # (A) 데이터 불러오기
    # -----------------------------------------------------------
    print("\n=== Tom&Jerry 데이터 로딩 중... ===")
    train_loader, val_loader, test_loader, class_to_idx = get_dataloaders(return_test=True)

    # -----------------------------------------------------------
    # (B) 모델 불러오기
    # -----------------------------------------------------------
    print(f"\n=== 모델 로딩: {MODEL_NAME} ===")
    model = get_model(MODEL_NAME).to(DEVICE)

    # -----------------------------------------------------------
    # (C) Optimizer — Stage1: 전체 Fine-Tuning
    # -----------------------------------------------------------
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # -----------------------------------------------------------
    # (D) Loss function
    # -----------------------------------------------------------
    criterion = get_loss()

    # -----------------------------------------------------------
    # (E) 실험 저장 폴더 생성
    # -----------------------------------------------------------
    run_dir = make_run_folder()
    ckpt_path = os.path.join(run_dir, f"{CKPT_NAME}.pth")

    # -----------------------------------------------------------
    # (F) 학습 기록 저장용 history dict
    # -----------------------------------------------------------
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "test_acc": [],
    }

    best_val_acc = 0.0

    # -----------------------------------------------------------
    # (G) Training Loop
    # -----------------------------------------------------------
    for epoch in range(1, NUM_EPOCHS + 1):

        ## Train
        train_loss, train_acc = train_one_epoch(
            epoch, model, train_loader, optimizer, criterion, DEVICE
        )

        ## Val
        val_loss, val_acc = eval_one_epoch(
            epoch, model, val_loader, criterion, DEVICE, tag="Val"
        )

        ## Test
        _, test_acc = eval_one_epoch(epoch, model, test_loader, criterion, DEVICE, tag="Test")

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["test_acc"].append(test_acc)

        # ----- Best checkpoint 저장 -----
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), ckpt_path)
            print(f"[Save!] Best 모델 저장! val_acc={val_acc:.4f}")

    print("\n=== Stage 1 Training Finished ===")
    print(f"Best Validation Accuracy = {best_val_acc:.4f}")

    # -----------------------------------------------------------
    # (H) 학습 곡선 그래프 저장
    # -----------------------------------------------------------
    print("\n=== 학습 그래프 저장 ===")
    plot_history(history, run_dir)

    # -----------------------------------------------------------
    # (I) Best 모델로 정밀 평가
    # -----------------------------------------------------------
    print("\n=== Best 모델 정밀 평가 ===")
    best_model = get_model(MODEL_NAME).to(DEVICE)
    best_model.load_state_dict(torch.load(ckpt_path))

    ## --- TEST SET에 대한 최종 성능 계산 ---
    y_true, y_pred, y_prob = gather_outputs(best_model, test_loader, DEVICE)
    metrics = compute_metrics(y_true, y_pred, y_prob)

    # 결과 출력
    print_metrics(metrics)

    # 메트릭 저장
    save_metrics_to_file(metrics, run_dir)

    # Confusion matrix 저장
    save_confusion_matrix(metrics["confusion_matrix"], EMOTIONS,
                          os.path.join(run_dir, "confusion_matrix_test.png"))


if __name__ == "__main__":
    main()
