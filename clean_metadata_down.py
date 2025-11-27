import os
import pandas as pd

BASE_ROOT = ".\preprocessed_final_data_down\Larva_preprocessed_down\train"
META_PATH = os.path.join(BASE_ROOT, ".\preprocessed_final_data_down\Larva_preprocessed_down\metadata_TL_test.csv")

df = pd.read_csv(META_PATH)

# 실제 파일 존재 여부 체크
df["exists"] = df["out_path"].apply(lambda p: os.path.exists(p))

total = len(df)
exist_cnt = df["exists"].sum()
missing_cnt = total - exist_cnt

print("총 샘플 수:", total)
print("실제 파일 있는 샘플 수:", exist_cnt)
print("실제 파일 없는 샘플 수:", missing_cnt)

# 혹시 어떤 애들이 빠졌는지 궁금하면 주석 풀고 확인
print("\n파일 없는 샘플 예시 5개:")
print(df[df["exists"] == False].head())

# 실제 파일이 있는 샘플만 남기기
df_clean = df[df["exists"] == True].copy()
df_clean = df_clean.drop(columns=["exists"])

out_path = os.path.join(BASE_ROOT, "metadata_down_clean_TL.csv")
df_clean.to_csv(out_path, index=False, encoding="utf-8-sig")

print("\n정리된 메타데이터 저장 완료:", out_path)