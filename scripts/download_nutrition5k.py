"""
Download and extract the Nutrition5k dataset.
Requires `gsutil` to be installed and in the PATH.
"""

import os
import subprocess
from pathlib import Path

def download():
    data_dir = Path("./data/nutrition5k")
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "imagery").mkdir(parents=True, exist_ok=True)
    
    print("Downloading Nutrition5k dataset metadata and imagery from Google Cloud Storage...")
    
    # The imagery can be huge (RGB + Depth + IR). We might just want the imagery/realsense/rgb
    # and the metadata.
    
    gsutil_path = "/home/drtex/.conda/envs/mlenv/bin/gsutil"
    commands = [
        # Download metadata
        [gsutil_path, "-m", "cp", "-r", "gs://nutrition5k_dataset/nutrition5k_dataset/metadata", str(data_dir)],
        # Download RGB images
        [gsutil_path, "-m", "cp", "-r", "gs://nutrition5k_dataset/nutrition5k_dataset/imagery/realsense_overhead", str(data_dir / "imagery")]
    ]
    
    for cmd in commands:
        print(f"Running: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed to download: {e}")
            print("Make sure gsutil is installed and configured if required.")
            return

    print(f"Nutrition5k dataset subset downloaded successfully to {data_dir}!")

if __name__ == "__main__":
    download()
