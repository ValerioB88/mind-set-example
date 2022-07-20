import os
import pathlib
import pickle

import numpy as np
import sty
import torch

from src.ebbinghaus.drawing_utils import DrawShape


class StaticDataEbbinghaus:
    """
    Static data:
    """
    def __init__(self, fun, size_dataset, path=None):

        print(sty.fg.yellow + f"Generating data in {path}" + sty.rs.fg)
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)

        self.labels = []
        self.images = []
        for N in range(size_dataset):
            img, rc = fun()
            img.save(path + f'/{rc/0.2}.png')


class EbbinghausRandomFlankers(StaticDataEbbinghaus):
    def __init__(self, img_size, *args, background='black',  **kwargs):
        img_size = (img_size, img_size)
        ds = DrawShape(background=background, img_size=img_size, resize_up_resize_down=True)

        def function():
            r_c = np.random.uniform(0.05, 0.2)
            img = ds.create_random_ebbinghaus(r_c=r_c, n=5, flankers_size_range=(0.05, 0.18), colour_center_circle=(255, 0, 0))
            return img, r_c
        super().__init__(function, *args, **kwargs)


class EbbinghausTestBigFlankers(StaticDataEbbinghaus):
    def __init__(self, img_size, *args, background='black',  **kwargs):
        img_size = (img_size, img_size)
        ds = DrawShape(background=background, img_size=img_size, resize_up_resize_down=True)
        n_large = 5

        def function():
            r_c = np.random.uniform(0.08, 0.1)
            r2 = np.random.uniform(0.12, 0.15)
            shift = np.random.uniform(0, np.pi)
            img = ds.create_ebbinghaus(r_c=r_c, d=0.02 + (r_c + r2), r2=r2, n=n_large, shift=shift, colour_center_circle=(255, 0, 0))
            return img, r_c
        super().__init__(function, *args, **kwargs)


class EbbinghausTestSmallFlankers(StaticDataEbbinghaus):
    def __init__(self, img_size, *args, background='black',  **kwargs):
        img_size = (img_size, img_size)
        ds = DrawShape(background=background, img_size=img_size, resize_up_resize_down=True)
        n_small = 8
        def function():
            r_c = np.random.uniform(0.08, 0.1)
            r2 = np.random.uniform(0.02, 0.08)
            shift = np.random.uniform(0, np.pi)
            img = ds.create_ebbinghaus(r_c=r_c, d=0.02 + (r_c + r2), r2=r2, n=n_small, shift=shift, colour_center_circle=(255, 0, 0))
            return img, r_c
        super().__init__(function, *args, **kwargs)


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--num_training_data', default=60000, type=int)
parser.add_argument('--num_testing_data', default=2000, type=int)
args = parser.parse_known_args()[0]
EbbinghausRandomFlankers(path=f'./data/ebbinghaus/train_random_data_{args.num_training_data}', size_dataset=6000, img_size=224, background='black')


EbbinghausRandomFlankers(path=f'./data/ebbinghaus/test_random_data_{args.num_testing_data}', size_dataset=2000, img_size=224, background='black')

EbbinghausTestSmallFlankers(path=f'./data/ebbinghaus/test_small_flankers_data_{args.num_testing_data}', size_dataset=2000, img_size=224, background='black')

EbbinghausTestBigFlankers(path=f'./data/ebbinghaus/test_big_flankers_data_{args.num_testing_data}', size_dataset=2000, img_size=224, background='black')