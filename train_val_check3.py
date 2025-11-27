import os
import pandas as pd

# 1. 메타데이터 로드
meta_path = os.path.join("Data/Dinocore_preprocessed", "metadata_all.csv")
df = pd.read_csv(meta_path)

# 2. TL / VL 타입 열 추가 (sequence 앞 두 글자)
df["seq_type"] = df["sequence"].str[:2]   # "TL", "VL", ...

# ===============================
# (A) Sadness 이미지 기준 TL/VL 비율
# ===============================

sad_df = df[df["emotion"] == "Sadness"].copy()

sad_img_counts = sad_df["seq_type"].value_counts()
sad_img_ratio = sad_img_counts / sad_img_counts.sum() * 100

print("=== [Sadness] 이미지 기준 TL/VL 개수 & 비율 ===")
print("\n[개수]")
print(sad_img_counts)
print("\n[비율 %]")
print(sad_img_ratio.round(2))

# ===============================
# (B) Sadness '시퀀스' 기준 TL/VL 비율
# ===============================

# 1) Sadness가 한 번이라도 등장하는 시퀀스만 추출
sad_seq = sad_df[["sequence", "seq_type"]].drop_duplicates()

sad_seq_counts = sad_seq["seq_type"].value_counts()
sad_seq_ratio_within_sad = sad_seq_counts / sad_seq_counts.sum() * 100

print("\n=== [Sadness] 시퀀스 기준 TL/VL 개수 & 비율 (Sadness 시퀀스만 중에서) ===")
print("\n[개수]")
print(sad_seq_counts)
print("\n[비율 %]")
print(sad_seq_ratio_within_sad.round(2))

# 2) TL 전체 시퀀스 중 Sadness를 포함하는 비율
all_seq = df[["sequence", "seq_type"]].drop_duplicates()
all_seq_counts = all_seq["seq_type"].value_counts()

# TL/VL별: Sadness 포함 시퀀스 / 전체 시퀀스
sad_seq_ratio_per_type = (sad_seq_counts / all_seq_counts * 100).round(2)

print("\n=== [Sadness] TL/VL 전체 시퀀스 중에서 Sadness를 포함하는 비율 ===")
print("(각 seq_type별: Sadness 시퀀스 수 / 전체 시퀀스 수)")
print(sad_seq_ratio_per_type)

# (옵션) CSV로 저장하고 싶으면:
out_dir = "Larva_preprocessed"
os.makedirs(out_dir, exist_ok=True)
sad_seq.to_csv(os.path.join(out_dir, "sadness_sequences_TL_VL.csv"),
               index=False, encoding="utf-8-sig")
print("\nSadness 포함 시퀀스 목록을 CSV로 저장했습니다:",
      os.path.join(out_dir, "sadness_sequences_TL_VL.csv"))
