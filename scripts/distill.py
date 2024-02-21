import argparse
import json
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from argparse import ArgumentParser
from pathlib import Path

import pytorch_lightning as pl
from configargparse import ArgumentParser as ConfigArgumentParser
from pytorch_lightning.loggers import TensorBoardLogger


from distillation.data import PanoheadPostDataModule
from distillation.model import NHAStaticTrainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run distillation from Panohead to FLAME")
    parser.add_argument("path", type=str, help="Path to folder with facial photos && 3DDFA_V2 data")
    args = parser.parse_args()

    path = Path(args.path).resolve().absolute()

    ds_path = path / "ds"
    pti_out_path = path / "pti_out"
    num_frames = len(list(ds_path.glob("frame_*")))

    # Generate full split.json
    with open("deps/neural-head-avatars/split.json", "w") as f:
        split = {
            "train": list(range(num_frames)),
            "val": list(range(num_frames)),
        }
        json.dump(split, f)

    os.chdir("deps/neural-head-avatars")
    # print(os.getcwd())
    # print(Path(".").resolve())
    # print(os.path.abspath("."))

    args = [
        "--config=" + "configs/optimize_avatar_mesh_guidance.ini",
        "--default_root_dir=" + str(path / "ds" / "results"),
        "--data_path=" + str(path / "ds"),
        "--pti_out_path=" + str(path / "pti_out"),
        "--load_threeddfa=" + str(path / "dataset.json"),
        "--gpus=" + "1",
    ]

    parser = ArgumentParser()
    parser = NHAStaticTrainer.add_argparse_args(parser)
    parser = PanoheadPostDataModule.add_argparse_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)

    parser = ConfigArgumentParser(parents=[parser], add_help=False)
    parser.add_argument("--config", required=True, is_config_file=True)
    parser.add_argument(
        "--checkpoint_file",
        type=str,
        required=False,
        default="",
        help="checkpoint to load model from",
    )

    args = parser.parse_args() if args is None else parser.parse_args(args)
    args.replace_sampler_ddp = False
    args.load_flame = False
    args.load_camera = False
    args.image_log_period = 5
    # args.flame_lr = list(map(lambda x: x * 5, args.flame_lr))

    args_dict = vars(args)

    nha_static = NHAStaticTrainer(**args_dict)
    data = PanoheadPostDataModule(**args_dict)
    data.setup()

    experiment_logger = TensorBoardLogger(args_dict["default_root_dir"], name="lightning_logs")
    trainer = pl.Trainer.from_argparse_args(
        args, callbacks=nha_static.callbacks, max_epochs=1000, logger=experiment_logger
    )

    trainer.fit(
        nha_static,
        train_dataloader=data.train_dataloader(batch_size=1),
        val_dataloaders=data.val_dataloader(batch_size=3),
    )
