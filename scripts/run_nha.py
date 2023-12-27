import argparse
import os
import subprocess
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Neural-Head-Avatars')
    parser.add_argument('path', type=str, help='Path to folder with facial photos && 3DDFA_V2 data')
    args = parser.parse_args()

    path = Path(args.path).absolute()

    # Get latest tracking result
    tracking_results_path = path / 'ds' / 'tracking_results'
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