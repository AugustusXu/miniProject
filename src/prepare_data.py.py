import os
from huggingface_hub import snapshot_download

def prepare_data():
    # 路径基于在项目根目录执行
    local_dir = "./data/sks_dog"
    os.makedirs(local_dir, exist_ok=True)
    print("Downloading target subject images (DreamBooth Dog Dataset)...")
    
    snapshot_download(
        repo_id="diffusers/dog-example",
        local_dir=local_dir,
        repo_type="dataset",
        ignore_patterns=[".gitattributes", "*.md"]
    )
    print(f"✅ Data ready at: {local_dir}")

if __name__ == "__main__":
    prepare_data()