import cv2
import os
from pathlib import Path

# Load 3DDFA
import yaml

cur_dir = os.getcwd()
proj_path = Path(__file__).parent.parent
cfg = yaml.load(
    open(proj_path / "deps/3DDFA_V2/configs/mb1_120x120.yml"),
    Loader=yaml.SafeLoader,
)

os.chdir(str(proj_path / "deps/3DDFA_V2"))
from TDDFA import TDDFA
from FaceBoxes import FaceBoxes


tddfa = TDDFA(gpu_mode="gpu", **cfg)
face_boxes = FaceBoxes()
os.chdir(str(cur_dir))


def get_lmks(frame):
    boxes = face_boxes(frame)
    param_lst, roi_box_lst = tddfa(frame, boxes)
    ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=False)
    return ver_lst[0]


def read_mp4(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames
