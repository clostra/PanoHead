import argparse
import os
import subprocess
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run PanoHead avatar generation')
    parser.add_argument('path', type=str, help='Path to folder with facial photos')
    args = parser.parse_args()

    path = Path(args.path).absolute()

    # Generate avatar
    subprocess.run([
        'python', 
        'projector_withseg.py', 
        '--target_img', str(path),
        '--network', 'models/easy-khair-180-gpc0.8-trans10-025000.pkl',
        '--outdir', str(path / 'pti_out'),
    ], check=True)

    # Generate mesh
    subprocess.run([
        'python', 
        'gen_videos_proj_withseg.py', 
        '--output', str(path / 'pti_out' / 'PTI_render' / 'post.mp4'),
        '--latent', str(path / 'pti_out' / 'projected_w.npz'),
        '--trunc', '0.7',
        '--network', str(path / 'pti_out' / 'fintuned_generator.pkl'),
        '--cfg', 'Head',
        '--shapes'
    ], check=True)