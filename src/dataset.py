import os
import argparse
import requests

DATASET_ENDPOINT = "datasets/HuggingFaceFW/fineweb-edu"
DATASET_PATH = "tree/main/sample/10BT"
ARTIFACTS_PATH = "downloads"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download FineWeb-Edu 100BT dataset shards")
    parser.add_argument("-n", "--num-files", type=int, default=1, help="Number of files to download (default: 1)")
    args = parser.parse_args()
    
    os.makedirs(ARTIFACTS_PATH, exist_ok=True)
    print(f"Dataset: https://huggingface.co/api/{DATASET_ENDPOINT}/{DATASET_PATH}")
    files = requests.get(f"https://huggingface.co/api/{DATASET_ENDPOINT}/{DATASET_PATH}", stream=True, timeout=30).json()
    print(f"Total number of shards: {len(files)}")
    print(f"Requested number of shards: {args.num_files}\n")
    
    if len(files) > args.num_files:
        files = files[:args.num_files]
    for i, file in enumerate(files):
        local_path = os.path.join(ARTIFACTS_PATH, os.path.basename(file["path"]))
        if os.path.exists(local_path):
            print(f"[{i + 1}/{args.num_files}] File {file["path"]} exists {ARTIFACTS_PATH}")
            continue
        
        print(f"[{i + 1}/{args.num_files}] Downloading {file["path"]} ")
        
        with requests.get(f"https://huggingface.co/{DATASET_ENDPOINT}/resolve/main/{file["path"]}", stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        
        print(f"[{i + 1}/{args.num_files}] Downloaded {file["path"]} to {ARTIFACTS_PATH}")
        