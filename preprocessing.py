import os
import json
import collections
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

def imread_unicode(path):
    """
    Windows에서 한글/특수문자 경로 이미지 로드를 위한 유틸.
    cv2.imread 대신 사용.
    """
    try:
        # np.fromfile은 유니코드 경로 지원
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"[에러] imread_unicode 실패: {path} / {e}")
        return None


def find_seg_json(label_dir):
    """해당 라벨 디렉토리에서 *_seg_en.json 파일 하나를 찾아 반환."""
    cand = [f for f in os.listdir(label_dir) if f.endswith("seg_en.json")]
    if not cand:
        return None
    # 여러 개 있으면 첫 번째 사용 (필요시 규칙 수정 가능)
    return os.path.join(label_dir, cand[0])


def make_mask_from_segmentation(segmentation, height, width):
    """
    COCO-style segmentation(list of polygons) -> binary mask (uint8, 0/255)
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    # segmentation: [ [x1,y1,...], [x1,y1,...], ... ]
    for seg in segmentation:
        pts = np.array(seg, dtype=np.float32).reshape(-1, 2)
        pts = np.round(pts).astype(np.int32)
        pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)
        if pts.shape[0] >= 3:  # 최소 3점 이상이어야 다각형
            cv2.fillPoly(mask, [pts], 255)
    return mask


def safe_name(name: str) -> str:
    """폴더 이름으로 쓰기 안전하게 간단 정리."""
    if name is None:
        return "Unknown"
    return str(name).replace(" ", "_")


def process_one_sequence(label_dir, image_dir, out_root, seq_name, records):
    """
    한 개 시퀀스(TL_xxx / TS_xxx 쌍)에 대해:
    - seg_en.json 읽고
    - 배경 제거 + bbox 크롭 + 256x256 리사이즈
    - 캐릭터/감정별로 저장
    - 메타데이터 records 리스트에 추가
    """
    seg_json_path = find_seg_json(label_dir)
    if seg_json_path is None:
        print(f"[경고] seg_en.json 없음, 스킵: {label_dir}")
        return

    with open(seg_json_path, encoding="utf-8") as f:
        data = json.load(f)

    images = {img["id"]: img for img in data["images"]}
    categories = {c["id"]: c["name"] for c in data["categories"]}

    annos_by_img = collections.defaultdict(list)
    for ann in data["annotations"]:
        annos_by_img[ann["image_id"]].append(ann)

    for img_id, annos in annos_by_img.items():
        img_info = images.get(img_id)
        if img_info is None:
            continue

        file_name = img_info["file_name"]
        img_path = os.path.join(image_dir, file_name)

        img = imread_unicode(img_path)
        if img is None:
            print(f"[경고] 이미지 로드 실패(Unicode): {img_path}")
            continue

        height, width = img.shape[:2]

        for ann in annos:
            emotion = ann.get("attribute", None)
            if emotion is None or emotion == "":
                # 감정 라벨 없는 경우 스킵
                continue

            cat_id = ann["category_id"]
            character = categories.get(cat_id, "Unknown")

            # 1) segmentation -> mask (이 annotation 에 해당하는 캐릭터만)
            segmentation = ann.get("segmentation", None)
            if not segmentation:
                continue
            mask = make_mask_from_segmentation(segmentation, height, width)

            # 2) 배경 제거
            masked_img = cv2.bitwise_and(img, img, mask=mask)

            # 3) bbox 기준 크롭
            x, y, bw, bh = ann["bbox"]
            x0 = max(int(x), 0)
            y0 = max(int(y), 0)
            x1 = min(int(x + bw), width)
            y1 = min(int(y + bh), height)
            if x1 <= x0 or y1 <= y0:
                continue

            crop = masked_img[y0:y1, x0:x1]
            if crop.size == 0:
                continue

            # 4) 256x256 리사이즈
            crop_resized = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_AREA)

            # 5) 저장 경로: out_root/캐릭터/감정/시퀀스/
            char_dir = safe_name(character)
            emo_dir = safe_name(emotion)
            out_dir = os.path.join(out_root, char_dir, emo_dir, seq_name)
            os.makedirs(out_dir, exist_ok=True)

            base = os.path.splitext(file_name)[0]
            out_fname = f"{base}_ann{ann['id']}.png"
            out_path = os.path.join(out_dir, out_fname)

            cv2.imwrite(out_path, crop_resized)

            records.append({
                "out_path": out_path,
                "sequence": seq_name,
                "orig_file": file_name,
                "ann_id": ann["id"],
                "character": character,
                "emotion": emotion,
                "image_id": img_id,
            })


def main():
    # 이 스크립트를 'Larva_data' 바로 위에서 실행한다고 가정:
    # PROJECT_ROOT/
    #   preprocess_larva.py  ← 여기
    #   Larva_data/
    #     라벨링데이터/
    #     원천데이터/
    project_root = os.path.dirname(os.path.abspath(__file__))
    larva_root = os.path.join(project_root, "Larva_data")

    label_root = os.path.join(larva_root, "라벨링데이터")
    image_root = os.path.join(larva_root, "원천데이터")

    out_root = os.path.join(project_root, "Larva_preprocessed")
    os.makedirs(out_root, exist_ok=True)

    # 라벨링데이터 내부의 TL_*, VL_* 디렉토리만 대상으로 처리
    all_seq_dirs = []
    for name in sorted(os.listdir(label_root)):
        full = os.path.join(label_root, name)
        if not os.path.isdir(full):
            continue
        if name.startswith(("TL_", "VL_")):
            all_seq_dirs.append(name)

    print(f"총 시퀀스 개수: {len(all_seq_dirs)}")

    records = []

    for seq_name in tqdm(all_seq_dirs, desc="Sequences"):
        label_dir = os.path.join(label_root, seq_name)

        # TL_xxx_Larva_S.. -> TS_xxx_Larva_S.. (또는 VL_ -> VS_)
        if seq_name.startswith("TL_"):
            img_seq_name = "TS_" + seq_name[3:]
        elif seq_name.startswith("VL_"):
            img_seq_name = "VS_" + seq_name[3:]
        else:
            img_seq_name = seq_name

        image_dir = os.path.join(image_root, img_seq_name)

        if not os.path.isdir(image_dir):
            print(f"[경고] 대응되는 이미지 디렉토리 없음: {image_dir}")
            continue

        process_one_sequence(label_dir, image_dir, out_root, seq_name, records)

    # 전체 메타데이터 저장
    if records:
        meta_path = os.path.join(out_root, "metadata_all.csv")
        df = pd.DataFrame(records)
        df.to_csv(meta_path, index=False, encoding="utf-8-sig")
        print(f"메타데이터 저장 완료: {meta_path}")
        print(f"총 샘플 수: {len(records)}")
    else:
        print("생성된 샘플이 없습니다. 경로/JSON 구조를 다시 확인하세요.")


if __name__ == "__main__":
    main()