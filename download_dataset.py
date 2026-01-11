import kagglehub
import shutil
import os

# Download dataset
path = kagglehub.dataset_download(
    "heesoo37/120-years-of-olympic-history-athletes-and-results"
)

print("Downloaded to:", path)

# Move files into your project data/raw folder
RAW_DIR = "data/raw"
os.makedirs(RAW_DIR, exist_ok=True)

for file in os.listdir(path):
    src = os.path.join(path, file)
    dst = os.path.join(RAW_DIR, file)
    shutil.copy(src, dst)
    print(f"Copied {file} -> {RAW_DIR}")