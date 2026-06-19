# Install dependencies: transformers, torch and huggingface_hub
# pip install transformers torch huggingface_hub tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download
import torch
import os
from tqdm import tqdm
import time

def download_qwen3():
    model_id = "Qwen/Qwen3-8B"
    local_dir = os.getenv("QWEN_BASE_MODEL", "./Qwen")  # Local save directory

    print(f"Starting model download with snapshot_download: {model_id}")
    print(f"Download target path: {os.path.abspath(local_dir)}")
    print("=" * 50)

    start_time = time.time()

    # Download the model files using snapshot_download, showing progress
    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        resume_download=True,  # Support resumable downloads
        # allow_patterns=["*.bin", "*.safetensors", "*.json", "*.txt", "*.model"],  # Download model files
        # ignore_patterns=["*.md", "*.h5", "*.msgpack"],  # Ignore unneeded files
        tqdm_class=tqdm,  # Use tqdm to show the progress bar
        local_files_only=False,  # Allow downloading from the network
    )

    end_time = time.time()
    download_time = end_time - start_time

    print("=" * 50)
    print(f"Model files downloaded! Total time: {download_time:.2f} seconds")

    # Verify the downloaded files
    if os.path.exists(local_dir):
        print(f"Model saved to: {os.path.abspath(local_dir)}")

        # Compute the total file size
        total_size = 0
        files = []
        for root, dirs, filenames in os.walk(local_dir):
            for filename in filenames:
                file_path = os.path.join(root, filename)
                file_size = os.path.getsize(file_path)
                total_size += file_size
                files.append((filename, file_size))

        print(f"Number of files downloaded: {len(files)}")
        print(f"Total file size: {total_size / (1024 * 1024):.2f} MB")
        print("\nMain files:")

        # Sort by file size and show the largest files
        files.sort(key=lambda x: x[1], reverse=True)
        for filename, file_size in files[:10]:
            size_mb = file_size / (1024 * 1024)
            print(f"  - {filename} ({size_mb:.2f} MB)")

    else:
        print("Download failed; directory does not exist")

if __name__ == "__main__":
    download_qwen3()
