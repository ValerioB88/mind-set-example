import os
import pathlib
import pickle
from pathlib import Path
from time import time

import numpy as np
import torch
import torchvision
from PIL import ImageStat
from sty import fg, rs
from torchvision import transforms as tf


def add_compute_stats(obj_class):
    class ComputeStatsUpdateTransform(obj_class):
        # This class basically is used for normalize Dataset Objects such as ImageFolder in order to be used in our more general framework
        def __init__(self, name_ds='dataset', add_PIL_transforms=None, add_tensor_transforms=None, num_image_calculate_mean_std=70, stats=None, save_stats_file=None, **kwargs):
            """

            @param add_tensor_transforms:
            @param stats: this can be a dict (previous stats, which will contain 'mean': [x, y, z] and 'std': [w, v, u],
                          a str "ImageNet", indicating the ImageNet stats,
                          a str pointing to a path to a pickle file, containing a dict with 'mean' and 'std'
                          None, indicating that stats are gonna be computed
                        In any case, the stats are gonna be added as a normalizing transform.
            @param save_stats_file: a path, indicating where to save the stats
            @param kwargs:
            """
            self.verbose = True
            print(fg.yellow + f"\n**Creating Dataset [" + fg.cyan + f"{name_ds}" + fg.yellow + "]**" + rs.fg)
            super().__init__(**kwargs)
            # self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

            if add_PIL_transforms is None:
                add_PIL_transforms = []
            if add_tensor_transforms is None:
                add_tensor_transforms = []

            self.transform = torchvision.transforms.Compose([*add_PIL_transforms, torchvision.transforms.ToTensor(), *add_tensor_transforms])

            self.name_ds = name_ds
            self.additional_transform = add_PIL_transforms
            self.num_image_calculate_mean_std = num_image_calculate_mean_std

            compute_stats = False

            if isinstance(stats, dict):
                self.stats = stats
                print(fg.red + f"Using precomputed stats: " + fg.cyan + f"mean = {self.stats['mean']}, std = {self.stats['std']}" + rs.fg)

            elif stats == 'ImageNet':
                self.stats = {}
                self.stats['mean'] = [0.491, 0.482, 0.44]
                self.stats['std'] = [0.247, 0.243, 0.262]
                print(fg.red + f"Using ImageNet stats: " + fg.cyan + f"mean = {self.stats['mean']}, std = {self.stats['std']}" + rs.fg)


            elif isinstance(stats, str):
                if os.path.isfile(stats):
                    self.stats = pickle.load(open(stats, 'rb'))
                    print(fg.red + f"Using stats from file [{Path(stats).name}]: " + fg.cyan + f"mean = {self.stats['mean']}, std = {self.stats['std']}" + rs.fg)
                    if stats == save_stats_file:
                        save_stats_file = None
                else:
                    print(fg.red + f"File [{Path(stats).name}] not found, stats will be computed." + rs.fg)
                    compute_stats = True

            if stats is None or compute_stats is True:
                self.stats = self.call_compute_stats()

            if save_stats_file is not None:
                print(f"Stats saved in {save_stats_file}")
                pathlib.Path(os.path.dirname(save_stats_file)).mkdir(parents=True, exist_ok=True)
                pickle.dump(self.stats, open(save_stats_file, 'wb'))

            normalize = torchvision.transforms.Normalize(mean=self.stats['mean'],
                                                         std=self.stats['std'])

            self.transform.transforms += [normalize]

        def call_compute_stats(self):
            return compute_mean_and_std_from_dataset(self, None, max_iteration=self.num_image_calculate_mean_std, verbose=self.verbose)



    return ComputeStatsUpdateTransform


class Stats(ImageStat.Stat):
    def __add__(self, other):
        return Stats(list(map(np.add, np.array(self.h) / 255, np.array(other.h) / 255)))


def compute_mean_and_std_from_dataset(dataset, dataset_path=None, max_iteration=100, data_loader=None, verbose=True):
    if max_iteration < 30:
        print(f'Max Iteration in Compute Mean and Std for dataset is lower than 30! This could create unrepresentative stats!') if verbose else None
    start = time()
    stats = {}
    transform_save = dataset.transform
    if data_loader is None:
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=64, num_workers=0)

    statistics = None
    c = 0
    stop = False
    while stop is False:
        for data, _, _ in data_loader:
            for b in range(data.shape[0]):
                if c % 10 == 9 and verbose:
                    print(f'{c}/{max_iteration}, m: {np.around(np.array(statistics.mean) / 255, 4)}, std: {np.around(np.array(statistics.stddev) / 255, 4)}')
                c += 1
                if statistics is None:
                    statistics = Stats(tf.ToPILImage()(data[b]))
                else:
                    statistics += Stats(tf.ToPILImage()(data[b]))
                if c > max_iteration:
                    stop = True
                    break
            if stop:
                break

    stats['time_one_iter'] = (time() - start) / max_iteration
    stats['mean'] = np.array(statistics.mean) / 255
    stats['std'] = np.array(statistics.stddev) / 255
    stats['iter'] = max_iteration
    print(fg.cyan + f'mean={np.around(stats["mean"], 4)}, std={np.around(stats["std"], 4)}, time1it: {np.around(stats["time_one_iter"], 4)}s' + rs.fg) if verbose else None

    if dataset_path is not None:
        print('Saving in {}'.format(dataset_path))
        with open(dataset_path, 'wb') as f:
            pickle.dump(stats, f)

    dataset.transform = transform_save
    return stats