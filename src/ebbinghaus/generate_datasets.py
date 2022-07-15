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
        # if path and os.path.isfile(path):
        #     print(sty.fg.yellow + f"Dataset Loaded from {path}" + sty.rs.fg)
        #     data = pickle.load(open(path, 'rb'))
        #     self.images, self.labels = data['images'], data['labels']
        # else:
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)

        self.labels = []
        self.images = []
        for N in range(size_dataset):
            img, rc = fun()
            img.save(path + f'/{rc/0.2}.png')
            # self.labels.append(rc/0.2)
            # self.images.append(img)
        # if path:

        # pickle.dump({'images': self.images,
        #              'labels': self.labels}, open(path, 'wb'))
        # print(sty.fg.yellow + f"Dataset written in {path}" + sty.rs.fg)
        # if not hasattr(self, 'transform'):
        #     self.transform = None


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



EbbinghausRandomFlankers(path='./data/ebbinghaus/train_random_data', size_dataset=60000, img_size=224, background='black')


EbbinghausRandomFlankers(path='./data/ebbinghaus/test_random_data', size_dataset=2000, img_size=224, background='black')

EbbinghausTestSmallFlankers(path='./data/ebbinghaus/test_small_flankers_data', size_dataset=2000, img_size=224, background='black')

EbbinghausTestBigFlankers(path='./data/ebbinghaus/test_big_flankers_data', size_dataset=2000, img_size=224, background='black')