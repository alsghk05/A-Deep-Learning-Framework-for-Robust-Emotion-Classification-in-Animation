import os
import shutil
from tqdm import tqdm

def flatten_directory(root_dir):
    """
    root_dir 내부의 모든 하위 디렉토리를 순회하며
    그 안에 있는 파일들을 root_dir로 꺼내(flatten) 이동시키는 함수.

    예:
    root_dir/
        TL_001/*.png
        TL_002/*.png

    → flatten 후:

    root_dir/
        img1.png
        img2.png
        img3.png
        ...
    """

    print(f"[INFO] Flatten 시작: {root_dir}")

    # root_dir 기준으로 하위 폴더 탐색
    for subdir, dirs, files in os.walk(root_dir):
        # root_dir 자체는 건너뜀 (flatten 대상 X)
        if subdir == root_dir:
            continue

        for filename in files:
            src = os.path.join(subdir, filename)
            dst = os.path.join(root_dir, filename)

            # 파일 이름 충돌 방지: 동일 파일명 있으면 뒤에 번호 붙이기
            if os.path.exists(dst):
                base, ext = os.path.splitext(filename)
                count = 1
                new_name = f"{base}_{count}{ext}"
                dst = os.path.join(root_dir, new_name)
                while os.path.exists(dst):
                    count += 1
                    new_name = f"{base}_{count}{ext}"
                    dst = os.path.join(root_dir, new_name)

            shutil.move(src, dst)

    print(f"[DONE] Flatten 완료: {root_dir}")


# ---------------------------
# 실행 예시
# ---------------------------
if __name__ == "__main__":
    # 원하는 디렉토리[1] 경로 입력
        target_dir = r"/dataset_Larva\Yellow\Anger"
        flatten_directory(target_dir)
