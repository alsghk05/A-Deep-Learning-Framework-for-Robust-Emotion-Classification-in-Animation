import os
import pandas as pd

# 1. 메타데이터 로드
meta_path = os.path.join("Dinocore_preprocessed", "metadata_all.csv")
df = pd.read_csv(meta_path)

# 2. TL / VL 타입 열 추가
#   sequence 예: "TL_094_Larva_S01_001", "VL_013_Larva_S01_090"
df["seq_type"] = df["sequence"].str[:2]  # "TL", "VL", ...

# 3. TL/VL × 캐릭터 × 감정별 개수 집계
group = (
    df.groupby(["seq_type", "character", "emotion"])
      .size()
      .reset_index(name="count")
)

# 4. 같은 (seq_type, character) 안에서 감정 비율(%) 계산
#    예: TL + Red 안에서 Happiness가 몇 %인지
group["ratio_in_char_%"] = (
    group.groupby(["seq_type", "character"])["count"]
         .transform(lambda x: x / x.sum() * 100)
)

# 보기 좋게 정렬
group = group.sort_values(["seq_type", "character", "emotion"])

print("=== TL / VL 별 · 캐릭터별 · 감정별 개수 & 비율(%) ===")
print(group)

# 5. (옵션) 피벗 테이블로 정리해서 보기: 개수 기준
pivot_count = group.pivot_table(
    index=["seq_type", "character"],  # 행: TL/VL + 캐릭터
    columns="emotion",               # 열: 감정
    values="count",
    fill_value=0
).astype(int)

print("\n=== 피벗 테이블 (개수) ===")
print(pivot_count)

# 6. (옵션) 피벗 테이블: 비율(%) 기준
pivot_ratio = group.pivot_table(
    index=["seq_type", "character"],
    columns="emotion",
    values="ratio_in_char_%",
    fill_value=0
).round(2)

print("\n=== 피벗 테이블 (비율 %) ===")
print(pivot_ratio)

# 7. (옵션) CSV로 저장
out_dir = "Larva_preprocessed"
os.makedirs(out_dir, exist_ok=True)

group.to_csv(os.path.join(out_dir, "dist_TL_VL_char_emo_long.csv"),
             index=False, encoding="utf-8-sig")

pivot_count.to_csv(os.path.join(out_dir, "dist_TL_VL_char_emo_count.csv"),
                   encoding="utf-8-sig")

pivot_ratio.to_csv(os.path.join(out_dir, "dist_TL_VL_char_emo_ratio.csv"),
                   encoding="utf-8-sig")

print("\nCSV 저장 완료:")
print(" - dist_TL_VL_char_emo_long.csv")
print(" - dist_TL_VL_char_emo_count.csv")
print(" - dist_TL_VL_char_emo_ratio.csv")
