import os
import zipfile

def extract_all_zips(base_dir):
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.zip'):
                zip_path = os.path.join(root, file)
                extract_dir = os.path.join(root, file.replace('.zip', ''))
                os.makedirs(extract_dir, exist_ok=True)
                
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                
                print(f"✅ Extracted: {zip_path} → {extract_dir}")

if __name__ == "__main__":
    base_dirs = [
        "./Larva_data/라벨링데이터",
        "./Larva_data/원천데이터",
        "./BingBing_data/라벨링데이터",
        "./BingBing_data/원천데이터",
        "./MBA_data/라벨링데이터",
        "./MBA_data/원천데이터",
        "./Dinocore_data/라벨링데이터",
        "./Dinocore_data/원천데이터",
    ]
    for d in base_dirs:
        extract_all_zips(d)
