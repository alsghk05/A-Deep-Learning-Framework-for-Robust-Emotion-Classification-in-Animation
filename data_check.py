import pandas as pd
import os

# 1. 메타데이터 로드 (경로는 상황에 맞게 조정)
meta_path = os.path.join("Dinocore_preprocessed", "metadata_all.csv")
df = pd.read_csv(meta_path)

print("총 샘플 수:", len(df))
print("=" * 60)

# 2. 캐릭터 × 감정별 개수 집계
char_emo_counts = (
    df.groupby(["character", "emotion"])
    .size()
    .reset_index(name="count")
    .sort_values(["character", "emotion"])
)

print("▶ 캐릭터 × 감정별 이미지 개수 (long format)")
print(char_emo_counts)
print("=" * 60)

# 3. 피벗 테이블 (행: 캐릭터, 열: 감정)
pivot = (
    char_emo_counts
    .pivot(index="character", columns="emotion", values="count")
    .fillna(0)
    .astype(int)
)

print("▶ 캐릭터 × 감정별 이미지 개수 (pivot table)")
print(pivot)
print("=" * 60)

# 4. 감정별 전체 개수
print("▶ 감정별 전체 이미지 개수")
print(df["emotion"].value_counts())
print("=" * 60)

# 5. 캐릭터별 전체 개수
print("▶ 캐릭터별 전체 이미지 개수")
print(df["character"].value_counts())
print("=" * 60)

# 6. (옵션) CSV로 저장해두기
out_csv_path = os.path.join("Dinocore_preprocessed", "label_distribution_char_emo.csv")
pivot.to_csv(out_csv_path, encoding="utf-8-sig")
print("캐릭터 × 감정 분포를 CSV로 저장했습니다:", out_csv_path)
