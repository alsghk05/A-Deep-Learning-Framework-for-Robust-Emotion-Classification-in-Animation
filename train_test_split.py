import os
import shutil
import pandas as pd
from math import ceil

# ------------------------
# 경로 설정
# ------------------------
SRC_ROOT = "Dinocore_preprocessed_down"  # 다운샘플된 데이터 루트
META_PATH = os.path.join(SRC_ROOT, "metadata_down.csv")

TRAIN_ROOT = os.path.join(SRC_ROOT, "train")
TEST_ROOT = os.path.join(SRC_ROOT, "test")

META_TRAIN = os.path.join(SRC_ROOT, "metadata_TL_train.csv")
META_TEST = os.path.join(SRC_ROOT, "metadata_TL_test.csv")

RANDOM_STATE = 42  # 재현성

# ------------------------
# 1. 메타데이터 로드
# ------------------------
df = pd.read_csv(META_PATH)

# TL / VL 구분 컬럼 (sequence 앞 두 글자)
df["seq_type"] = df["sequence"].str[:2]

# TL만 사용
df_tl = df[df["seq_type"] == "TL"].copy()
print("TL 샘플 수:", len(df_tl))

# ------------------------
# 2. 시퀀스 단위로 9:1 split
# ------------------------
# TL 시퀀스 목록
seqs = df_tl["sequence"].drop_duplicates().sort_values().tolist()
n_seq = len(seqs)
print("TL 시퀀스 개수:", n_seq)

# 9:1 split → train 시퀀스 개수
n_train_seq = int(round(n_seq * 0.9))
n_train_seq = max(1, min(n_seq - 1, n_train_seq))  # 최소 1개 test 보장

print(f"train 시퀀스: {n_train_seq}개, test 시퀀스: {n_seq - n_train_seq}개")

# 섞어서 시퀀스 나누기
seqs_shuffled = pd.Series(seqs).sample(frac=1.0, random_state=RANDOM_STATE).tolist()
train_seqs = set(seqs_shuffled[:n_train_seq])
test_seqs = set(seqs_shuffled[n_train_seq:])

# train/test 데이터프레임 분리
df_train = df_tl[df_tl["sequence"].isin(train_seqs)].copy()
df_test = df_tl[df_tl["sequence"].isin(test_seqs)].copy()

print("train 샘플 수:", len(df_train))
print("test 샘플 수:", len(df_test))

# 감정/캐릭터 분포 간단 확인
print("\n[train] emotion 분포")
print(df_train["emotion"].value_counts())
print("\n[test] emotion 분포")
print(df_test["emotion"].value_counts())

print("\n[train] character 분포")
print(df_train["character"].value_counts())
print("\n[test] character 분포")
print(df_test["character"].value_counts())

# ------------------------
# 3. 파일 복사 (train/test 폴더 생성)
# ------------------------
def copy_split(df_split, dst_root):
    for row in df_split.itertuples(index=False):
        src_path = row.out_path

        # SRC_ROOT 기준 상대경로 계산
        # (metadata_down에서 out_path가 절대경로/상대경로 무엇이든
        #  SRC_ROOT 아래에만 있으면 relpath로 잘라서 재구성 가능)
        rel_path = os.path.relpath(os.path.abspath(src_path), os.path.abspath(SRC_ROOT))
        dst_path = os.path.join(dst_root, rel_path)

        dst_dir = os.path.dirname(dst_path)
        os.makedirs(dst_dir, exist_ok=True)

        if not os.path.exists(src_path):
            print(f"[경고] 원본 이미지 없음, 스킵: {src_path}")
            continue

        shutil.copy2(src_path, dst_path)

        # out_path를 새로운 위치 기준으로 수정
        # (나중에 metadata_TL_train/test에서 사용)
        # DataFrame에는 반영하지 않고, 나중에 별도 컬럼 업데이트
    print(f"{dst_root} 로 복사 완료")


# 먼저 train/test 폴더 깨끗하게 만들고 싶으면 주석 해제
# if os.path.exists(TRAIN_ROOT):
#     shutil.rmtree(TRAIN_ROOT)
# if os.path.exists(TEST_ROOT):
#     shutil.rmtree(TEST_ROOT)

print("\n[복사 시작] train")
copy_split(df_train, TRAIN_ROOT)

print("\n[복사 시작] test")
copy_split(df_test, TEST_ROOT)

# ------------------------
# 4. metadata 내 out_path를 train/test 기준으로 업데이트
# ------------------------
def update_out_path(row, split_root):
    src_path = row["out_path"]
    rel_path = os.path.relpath(os.path.abspath(src_path), os.path.abspath(SRC_ROOT))
    return os.path.join(split_root, rel_path)

df_train["out_path"] = df_train.apply(lambda r: update_out_path(r, TRAIN_ROOT), axis=1)
df_test["out_path"] = df_test.apply(lambda r: update_out_path(r, TEST_ROOT), axis=1)

# ------------------------
# 5. train/test 메타데이터 저장
# ------------------------
df_train.to_csv(META_TRAIN, index=False, encoding="utf-8-sig")
df_test.to_csv(META_TEST, index=False, encoding="utf-8-sig")

print("\n메타데이터 저장 완료:")
print(" -", META_TRAIN)
print(" -", META_TEST)

print("\n작업 완료! 🎉")
