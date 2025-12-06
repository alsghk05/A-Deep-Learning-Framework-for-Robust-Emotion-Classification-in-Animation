import os
import glob
import random
from typing import List, Dict, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from configs.config_TomJerry import (
    TRAIN_DIR, VAL_DIR, TEST_DIR,
    EMOTIONS, BATCH_SIZE, NUM_WORKERS
)

# ===========================================================
# Emotion Dataset Class
# ===========================================================
class EmotionDataset(Dataset):
    """
    root/**/<emotion>/*.png
    """

    # --------------------------------------
    # def 1: root_dir, emotions, transform
    # --------------------------------------
    def __init__(self, root_dir: str, emotions: List[str], transform=None, shuffle_samples=True):
        self.root_dir = root_dir
        self.emotions = emotions
        self.transform = transform

        ## --- emotion → index ---
        self.class_to_idx: Dict[str, int] = {
            emo: i for i, emo in enumerate(emotions)
        }

        ## --- (img_path, label) 리스트 ---
        self.samples: List[Tuple[str, int]] = []

        ## --- 감정별 이미지 로드 ---
        for emo in emotions:
            label = self.class_to_idx[emo]

            patterns = [
                os.path.join(root_dir, "*", emo, "*.png"),
                os.path.join(root_dir, "*", emo, "*.jpg"),
                os.path.join(root_dir, "*", emo, "*.jpeg"),
            ]

            emo_paths = []
            for p in patterns:
                emo_paths.extend(glob.glob(p))

            ## --- Remove duplicates ---
            emo_paths = list(set(emo_paths))

            ## --- Save ---
            # ex. (".../Anger/Jerry_Anger_0001.png", 0),
            # ex. (".../Happiness/Jerry_Happy_0001.png", 1)
            for path in emo_paths:
                self.samples.append((path, label))

        ## --- Sample Shuffle ---
        if shuffle_samples:
            random.shuffle(self.samples)

        print(f"[EmotionDataset] root={root_dir}, 감정 샘플 수={len(self.samples)}")

    # --------------------------------------
    # def 2: return length of dataset
    # --------------------------------------
    def __len__(self):
        return len(self.samples)

    # --------------------------------------
    # def 3: index로 개별 샘플 로드
    # --------------------------------------
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")  # PNG RGBA → RGB

        if self.transform:
            img = self.transform(img)

        return img, label


# ===========================================================
# Transforms
# ===========================================================
def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    return train_transform, val_transform


# ===========================================================
# get_dataloaders() — train, val, test 모두 반환
# ===========================================================
def get_dataloaders(return_test=False):

    train_transform, val_transform = get_transforms()

    train_dataset = EmotionDataset(TRAIN_DIR, EMOTIONS, transform=train_transform)
    val_dataset   = EmotionDataset(VAL_DIR, EMOTIONS, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    ##
    if return_test:
        test_dataset = EmotionDataset(TEST_DIR, EMOTIONS, transform=val_transform)

        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True
        )

        return train_loader, val_loader, test_loader, train_dataset.class_to_idx

    ## 기본값(return_test=False)일 때
    return train_loader, val_loader, train_dataset.class_to_idx
