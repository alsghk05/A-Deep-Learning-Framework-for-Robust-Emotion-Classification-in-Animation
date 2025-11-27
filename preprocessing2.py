import os
import shutil
import pandas as pd

# ------------------------
# 설정값
# ------------------------
SRC_ROOT = "Dinocore_preprocessed"          # 원본 전처리 폴더
DST_ROOT = "Dinocore_preprocessed_down"     # 다운샘플링 결과 저장 폴더

META_SRC = os.path.join(SRC_ROOT, "metadata_all.csv")
META_DST = os.path.join(DST_ROOT, "metadata_down.csv")

TARGET_EMOTIONS = ["Anger", "Happiness", "Sadness", "Surprise"]
TARGET_IMAGES_TL = 310   # 각 감정별 TL 이미지 수
TARGET_IMAGES_VL = 76    # 각 감정별 VL 이미지 수

RANDOM_STATE = 42        # 재현성을 위한 랜덤 시드

# ------------------------
# 1. 메타데이터 로드
# ------------------------
df = pd.read_csv(META_SRC)

# seq_type(TL/VL) 열 추가
df["seq_type"] = df["sequence"].str[:2]

# TL, VL 중 필요한 감정만 필터링
df = df[
    df["emotion"].isin(TARGET_EMOTIONS)
    & df["seq_type"].isin(["TL", "VL"])
].copy()

print("필터링 후 샘플 수:", len(df))

# ------------------------
# 2. 감정별 · TL/VL별로 다운샘플링
# ------------------------
down_list = []

for emo in TARGET_EMOTIONS:
    df_emo = df[df["emotion"] == emo]

    print(f"\n=== Emotion: {emo} ===")

    for seq_type, target_n in [("TL", TARGET_IMAGES_TL), ("VL", TARGET_IMAGES_VL)]:
        df_et = df_emo[df_emo["seq_type"] == seq_type]

        cur_n = len(df_et)
        if cur_n == 0:
            print(f"  - {seq_type}: 이미지가 0장이라 스킵합니다.")
            continue

        if cur_n <= target_n:
            # 이미지 수가 타깃보다 적으면 모두 사용
            chosen = df_et.copy()
            print(f"  - {seq_type}: {cur_n}장 (<= {target_n}), 전부 사용")
        else:
            # 랜덤 샘플링으로 target_n 장 선택
            chosen = df_et.sample(n=target_n, random_state=RANDOM_STATE)
            print(f"  - {seq_type}: {cur_n}장 중 {target_n}장 샘플링")

        down_list.append(chosen)

# 다운샘플링된 전체 데이터프레임
if len(down_list) == 0:
    print("다운샘플링된 샘플이 없습니다. 조건을 다시 확인하세요.")
    exit()

df_down = pd.concat(down_list, ignore_index=True)

print("\n다운샘플링 후 전체 샘플 수:", len(df_down))

# ------------------------
# 3. TL/VL 시퀀스 비율 확인 (참고용 출력)
# ------------------------
print("\n=== [참고] 다운샘플 후 TL/VL 시퀀스 통계 (4개 감정만) ===")

# 감정별, seq_type별 이미지 수
img_counts = df_down.groupby(["emotion", "seq_type"]).size()
print("\n[감정별 · TL/VL별 이미지 수]")
print(img_counts)

# 감정별, seq_type별 시퀀스 수
seq_counts = (
    df_down[["emotion", "seq_type", "sequence"]]
    .drop_duplicates()
    .groupby(["emotion", "seq_type"])
    .size()
)

print("\n[감정별 · TL/VL별 시퀀스 수]")
print(seq_counts)

# ------------------------
# 4. 파일 복사: Larva_preprocessed_down으로
# ------------------------
print("\n이미지 파일 복사 중...")

for row in df_down.itertuples(index=False):
    src_path = row.out_path  # 예: Larva_preprocessed/Red/Happiness/...
    # SRC_ROOT 기준 상대 경로 계산
    rel_path = os.path.relpath(src_path, SRC_ROOT)
    dst_path = os.path.join(DST_ROOT, rel_path)

    dst_dir = os.path.dirname(dst_path)
    os.makedirs(dst_dir, exist_ok=True)

    if not os.path.exists(src_path):
        print(f"[경고] 원본 이미지 없음, 스킵: {src_path}")
        continue

    shutil.copy2(src_path, dst_path)

# out_path를 다운샘플 폴더 기준으로 업데이트
df_down["out_path"] = df_down["out_path"].apply(
    lambda p: os.path.join(
        DST_ROOT, os.path.relpath(p, SRC_ROOT)
    )
)

# ------------------------
# 5. 다운샘플 metadata 저장
# ------------------------
os.makedirs(DST_ROOT, exist_ok=True)
df_down.to_csv(META_DST, index=False, encoding="utf-8-sig")
print("\n다운샘플 메타데이터 저장 완료:", META_DST)

print("\n작업 완료! 🎉")
