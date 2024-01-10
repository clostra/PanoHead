import argparse
import os
import subprocess
from pathlib import Path
from glob import glob
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Neural-Head-Avatars')
    parser.add_argument('path', type=str, help='Path to folder with facial photos && 3DDFA_V2 data')
    args = parser.parse_args()

    path = Path(args.path).absolute()

    ds_path = path / 'ds'
    num_frames = len(list(ds_path.glob('frame_*')))

    # Generate full split.json
    with open('deps/neural-head-avatars/split.json', 'w') as f:
        split = {
            'train': list(range(num_frames)),
            'val': list(range(num_frames)),
        }
        json.dump(split, f)

    # Get latest tracking result
    tracking_results_path = ds_path / 'tracking_results'
    tracking_results_list = list(tracking_results_path.iterdir())
    tracking_results_list.sort()
    subprocess.run([
        'python', 
        'python_scripts/optimize_nha.py', 
        '--config', 'configs/optimize_avatar_mesh_guidance.ini', 
        '--default_root_dir', str(path / 'ds' / 'results'),
        '--tracking_results_path', str(tracking_results_list[-1] / 'tracked_flame_params.npz'),
        '--data_path', str(path / 'ds'),
        '--gpus', '1',
        '--load_threeddfa', str(path / 'dataset.json'),
        '--mesh', str(path / 'pti_out' / 'PTI_render' / 'post_mesh_level2' / 'mesh.obj')
    ], cwd="deps/neural-head-avatars", check=True)