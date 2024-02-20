import argparse
import os
import subprocess
from pathlib import Path
import numpy as np
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Video-Head-Tracker')
    parser.add_argument('path', type=str, help='Path to folder with facial photos && 3DDFA_V2 data')
    # mesh_guidance - use 3DDFA_V2 landmark data and mesh to guide the tracking, using initial RGB photos as reference
    parser.add_argument('--mode', type=str, default='mesh_guidance', help='Mode of dataset construction', choices=['mesh_guidance', 'rgb_renders'])
    parser.add_argument('-c', '--convert', action='store_true', help='Convert video to dataset')
    args = parser.parse_args()

    path = Path(args.path).absolute()

    if args.mode == 'mesh_guidance':
        if args.convert:
            subprocess.run([
                'python', 
                'python_scripts/video_to_dataset.py', 
                '--video', str(path),
                '--out_path', str(path / 'ds')
            ], cwd="deps/neural-head-avatars", check=True)

        num_frames = len(list((path / 'ds').glob('frame_*')))

        cmd_args = [
            '--config', 'configs/tracking_person_0000.ini', 
            '--data_path', str(path / 'ds'),
            '--output_path', str(path / 'ds' / 'tracking_results'),
            '--initialize', 'triangulate',
            *[f'--keyframes={i}' for i in range(1, num_frames)],
            '--w_photo', '0',
            '--threeddfa_dataset_json', str(path / 'dataset.json')
        ]

        # Move 3DDFA_v2 helper files
        subprocess.run([
            'python', 
            'vht/optimize_tracking.py', 
            *cmd_args
        ], cwd="deps/neural-head-avatars/deps/video-head-tracker", check=True)
    elif args.mode == 'rgb_renders':
        if args.convert:
            subprocess.run([
                'python', 
                'python_scripts/video_to_dataset.py', 
                '--video', str(path / 'pti_out' / 'PTI_render' / 'post.mp4'),
                '--out_path', str(path / 'ds')
            ], cwd="deps/neural-head-avatars", check=True)

        # Need to convert the post_c.npy to a threeddfa dataset.json type file
        post_c = np.load(path / 'pti_out' / 'PTI_render' / 'post_c.npy')
        dataset_json = {
            f'frame_{i:04d}.png': c.tolist()
            for i, c in enumerate(post_c)
        }
        with open(str(path / 'pti_out' / 'PTI_render' / 'post_c_dataset.json'), 'w') as f:
            json.dump(dataset_json, f)

        cmd_args = [
            '--config', 'configs/tracking_person_0000.ini', 
            '--data_path', str(path / 'ds'),
            '--output_path', str(path / 'ds' / 'tracking_results'),
            '--initialize', 'triangulate',
            '--w_photo', '0',
            '--threeddfa_dataset_json', str(path / 'pti_out' / 'PTI_render' / 'post_c_dataset.json')
        ]

        # Move 3DDFA_v2 helper files
        subprocess.run([
            'python', 
            'vht/optimize_tracking.py', 
            *cmd_args
        ], cwd="deps/neural-head-avatars/deps/video-head-tracker", check=True)
    else:
        raise ValueError(f'Unknown mode: {args.mode}')