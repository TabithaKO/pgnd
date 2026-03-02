from typing import Union, Dict, Optional
from pathlib import Path
from omegaconf import DictConfig
import sys
import shutil
import numpy as np
import wandb
import warp as wp


Tape = wp.Tape

class CondTape(object):
    def __init__(self, tape: Optional[Tape], cond: bool = True) -> None:
        self.tape = tape
        self.cond = cond

    def __enter__(self):
        if self.tape is not None and self.cond:
            self.tape.__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        if self.tape is not None and self.cond:
            self.tape.__exit__(exc_type, exc_value, traceback)


def cfg2dict(cfg: DictConfig) -> Dict:
    """
    Recursively convert OmegaConf to vanilla dict
    :param cfg:
    :return:
    """
    cfg_dict = {}
    for k, v in cfg.items():
        if type(v) == DictConfig:
            cfg_dict[k] = cfg2dict(v)
        else:
            cfg_dict[k] = v
    return cfg_dict


class Logger:

    def __init__(self, cfg, project='pgnd-train', entity='tabby-research'):
        wandb.init(project=project, entity=entity, name=cfg.train.name)
        wandb.config = cfg2dict(cfg)
    
    def add_scalar(self, tag, scalar, step=None):
        wandb.log({tag: scalar}, step=step)

    def add_image(self, tag, img, step=None, scale=True):
        if scale:
            img = (img - img.min()) / (img.max() - img.min())
        wandb.log({tag: wandb.Image(img)}, step=step)

    def add_video(self, tag, video, step=None):
        wandb.log({tag: wandb.Video(video)}, step=step)


def mkdir(path: Path, resume=False, overwrite=False) -> None:
    while True:
        if overwrite:
            if path.is_dir():
                print('overwriting directory ({})'.format(path))
            shutil.rmtree(path, ignore_errors=True)
            path.mkdir(parents=True, exist_ok=True)
            return
        elif resume:
            print('resuming directory ({})'.format(path))
            path.mkdir(parents=True, exist_ok=True)
            return
        else:
            if path.exists():
                feedback = input('target directory ({}) already exists, overwrite? [Y/r/n] '.format(path))
                ret = feedback.casefold()
            else:
                ret = 'y'
            if ret == 'n':
                sys.exit(0)
            elif ret == 'r':
                resume = True
            elif ret == 'y':
                overwrite = True


def get_root(path: Union[str, Path], name: str = '.root') -> Path:
    root = Path(path).resolve()
    while not (root / name).is_file():
        root = root.parent
    return root
