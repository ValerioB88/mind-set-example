import os
import pathlib
import pickle

import numpy as np
import sty
import torch

from src.ebbinghaus_illusion.generate_illusion.drawing_utils import DrawShape


class StaticDataEbbinghaus:
    def __init__(self, fun, size_dataset, path=None):
        if path and os.path.isfile(path):
            print(sty.fg.yellow + f"Dataset Loaded from {path}" + sty.rs.fg)
            data = pickle.load(open(path, 'rb'))
            self.images, self.labels = data['images'], data['labels']
        else:

            self.labels = []
            self.images = []
            for N in range(size_dataset):
                img, rc = fun()
                self.labels.append(rc)
                self.images.append(img)
            if path:
                pathlib.Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)

                pickle.dump({'images': self.images,
                             'labels': self.labels}, open(path, 'wb'))
                print(sty.fg.yellow + f"Dataset written in {path}" + sty.rs.fg)
        if not hasattr(self, 'transform'):
            self.transform = None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.transform(self.images[idx]) if self.transform else self.images, \
               torch.tensor(self.labels[idx])


class EbbinghausTrain(StaticDataEbbinghaus):
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