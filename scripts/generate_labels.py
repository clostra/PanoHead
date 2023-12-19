import argparse
import os
import subprocess
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate 3DDFA labels')
    parser.add_argument('path', type=str, help='Path to folder with facial photos')
    args = parser.parse_args()

    path = Path(args.path)

    # Move 3DDFA_v2 helper files
    subprocess.run([
        "cp", 
        "3DDFA_V2_cropping/dlib_kps.py", 
        "3DDFA_V2_cropping/recrop_images.py", 
        "deps/3DDFA_V2"
    ], check=True)

    tddfa_path = Path("deps/3DDFA_V2")
    target_img_path = tddfa_path / "test"
    subprocess.run([
        "rm",
        "-r", 
        str(target_img_path)
    ], check=True)
    os.makedirs(target_img_path, exist_ok=True)
    subprocess.run([
        "cp",
        "-r", 
        str(path), 
        str(target_img_path / "original")
    ], check=True)

    subprocess.run([
        "python", 
        "dlib_kps.py"
    ], cwd=str(tddfa_path), check=True)

    subprocess.run([
        "python", 
        "recrop_images.py", 
        "-i", 
        "data.pkl", 
        "--out_dir",
        str(path.absolute()),
        "-j", 
        "dataset.json"
    ], cwd=str(tddfa_path), check=True)