# emotion_dataset.py

import os
import glob
from typing import List, Dict, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from config import TRAIN_DIR, VAL_DIR, EMOTIONS, BATCH_SIZE, NUM_WORKERS


class EmotionDataset(Dataset):
    """
    디렉토리 구조 예:
      train/
        Emily/
          Anger/
            TL_xxx/
              *.png
          Happiness/
        Rex/
          Anger/
          ...
    root_dir 아래에서 emotions 이름이 포함된 폴더를 재귀적으로 검색해서
    이미지들을 수집하고, 각 emotion을 라벨로 사용.
    """

    def __init__(self, root_dir: str, emotions: List[str], transform=None):
        self.root_dir = root_dir
        self.emotions = emotions
        self.transform = transform

        self.class_to_idx: Dict[str, int] = {
            emo: i for i, emo in enumerate(emotions)
        }
        self.samples: List[Tuple[str, int]] = []

        for emo in emotions:
            label = self.class_to_idx[emo]

            # 예: root/**/Anger/**/*.png, jpg, jpeg
            patterns = [
                os.path.join(root_dir, "**", emo, "**", "*.png"),
                os.path.join(root_dir, "**", emo, "**", "*.jpg"),
                os.path.join(root_dir, "**", emo, "**", "*.jpeg"),
            ]

            emo_paths = []
            for p in patterns:
                emo_paths.extend(glob.glob(p, recursive=True))

            for path in emo_paths:
                self.samples.append((path, label))

        print(f"[EmotionDataset] root={root_dir}, 샘플 수={len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")  # PNG RGBA → RGB

        if self.transform is not None:
            img = self.transform(img)

        return img, label


def get_transforms():
    """학습/검증용 transform을 리턴"""
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return train_transform, val_transform


def get_dataloaders():
    """
    train_loader, val_loader, class_to_idx 반환
    (train/test 디렉토리는 config.py에서 가져옴)
    """
    train_transform, val_transform = get_transforms()

    train_dataset = EmotionDataset(TRAIN_DIR, EMOTIONS, transform=train_transform)
    val_dataset   = EmotionDataset(VAL_DIR,   EMOTIONS, transform=val_transform)

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

    return train_loader, val_loader, train_dataset.class_to_idx
