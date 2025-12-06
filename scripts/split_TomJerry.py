import os
import glob
import random
import shutil

# ======================================
# Config
# ======================================
SOURCE_DIR = r"C:\Users\ei994\PycharmProjects\DL_TP\dataset_TomJerry"        # 감정 폴더가 있는 디렉토리
OUT_DIR = r"C:\Users\ei994\PycharmProjects\DL_TP\dataset_TomJerry_split"     # 최종 train/val 저장 경로

TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

EMOTIONS = ["Anger", "Happiness", "Sadness", "Surprise"]
CHARACTERS = ["Jerry", "Tom"]  # 캐릭터 유지하기 위해 명시


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def split_character_emotion_folder(character, emotion):

    src_folder = os.path.join(SOURCE_DIR, character, emotion)

    files = glob.glob(os.path.join(src_folder, "*.png")) + \
            glob.glob(os.path.join(src_folder, "*.jpg")) + \
            glob.glob(os.path.join(src_folder, "*.jpeg"))

    files = list(set(files))

    random.shuffle(files)

    total = len(files)

    train_count = int(total * TRAIN_RATIO)
    val_count = int(total * VAL_RATIO)
    test_count = total - train_count - val_count

    train_files = files[:train_count]
    val_files   = files[train_count : train_count + val_count]
    test_files  = files[train_count + val_count :]

    print(f"[{character}/{emotion}] total={total}, train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")

    # 저장 경로 생성
    train_dst = os.path.join(OUT_DIR, "train", character, emotion)
    val_dst   = os.path.join(OUT_DIR, "val", character, emotion)
    test_dst  = os.path.join(OUT_DIR, "test", character, emotion)

    ensure_dir(train_dst)
    ensure_dir(val_dst)
    ensure_dir(test_dst)

    # 파일 복사
    for f in train_files:
        shutil.copy(f, os.path.join(train_dst, os.path.basename(f)))

    for f in val_files:
        shutil.copy(f, os.path.join(val_dst, os.path.basename(f)))

    for f in test_files:
        shutil.copy(f, os.path.join(test_dst, os.path.basename(f)))


if __name__ == "__main__":
    print("===== Tom & Jerry Character-Preserved 7:2:1 Split 시작 =====")

    # 캐릭터별 × 감정별 split
    for ch in CHARACTERS:
        for emo in EMOTIONS:
            split_character_emotion_folder(ch, emo)

    print("\n[DONE] dataset_TomJerry_split 생성 완료!")
