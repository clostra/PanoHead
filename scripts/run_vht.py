import argparse
import os
import subprocess
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Video-Head-Tracker')
    parser.add_argument('path', type=str, help='Path to folder with facial photos && 3DDFA_V2 data')
    # boolean flag
    parser.add_argument('-c', '--convert', action='store_true', help='Convert video to dataset')
    args = parser.parse_args()

    path = Path(args.path).absolute()

    if args.convert:
        subprocess.run([
            'python', 
            'python_scripts/video_to_dataset.py', 
            '--video', str(path),
            '--out_path', str(path / 'ds')
        ], cwd="deps/neural-head-avatars", check=True)

    cmd_args = [
        '--config', 'configs/tracking_person_0000.ini', 
        '--data_path', str(path / 'ds'),
        '--output_path', str(path / 'ds' / 'tracking_results'),
        '--initialize', 'triangulate',
        '--w_photo', '0',
        '--threeddfa_dataset_json', str(path / 'dataset.json')
    ]

    # Move 3DDFA_v2 helper files
    subprocess.run([
        'python', 
        'vht/optimize_tracking.py', 
        *cmd_args
    ], cwd="deps/neural-head-avatars/deps/video-head-tracker", check=True)