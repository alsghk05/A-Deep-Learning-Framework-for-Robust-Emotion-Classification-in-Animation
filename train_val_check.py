import os
import pandas as pd

meta_path = os.path.join("Dinocore_preprocessed", "metadata_all.csv")
df = pd.read_csv(meta_path)

# sequence 컬럼에서 TL_/VL_ prefix 추출
df["seq_type"] = df["sequence"].str[:2]  # "TL", "VL" 등

print("=== 이미지 기준 TL / VL 개수 & 비율 ===")
img_counts = df["seq_type"].value_counts()
img_ratio = df["seq_type"].value_counts(normalize=True) * 100

print("\n[이미지 개수]")
print(img_counts)
print("\n[이미지 비율(%)]")
print(img_ratio.round(2))

# 시퀀스별로도 계산
seq_df = df[["sequence"]].drop_duplicates().copy()
seq_df["seq_type"] = seq_df["sequence"].str[:2]

print("\n=== 시퀀스 기준 TL / VL 개수 & 비율 ===")
seq_counts = seq_df["seq_type"].value_counts()
seq_ratio = seq_df["seq_type"].value_counts(normalize=True) * 100

print("\n[시퀀스 개수]")
print(seq_counts)
print("\n[시퀀스 비율(%)]")
print(seq_ratio.round(2))