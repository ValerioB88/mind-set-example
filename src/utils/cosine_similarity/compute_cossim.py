import sty
from torchvision.transforms.functional import InterpolationMode
import re
import glob
import pandas as pd
from src.utils.net_utils import GrabNet, prepare_network, make_cuda
from src.utils.cosine_similarity.activation_recorder import RecordActivations
from src.utils.misc import conver_tensor_to_plot
import matplotlib.pyplot as plt
import numpy as np
from sty import fg, rs
import pickle
import torch
import os
import pathlib
from tqdm import tqdm
import torchvision.transforms as transforms
from copy import deepcopy
import torchvision
from torchvision.transforms import functional as F
import PIL.Image as Image


def save_figs(path, set, extra_info='', n=None):
    fig, ax = plt.subplots(len(set) if n is None else n, 2)
    if np.ndim(ax) == 1:
        ax = np.array([ax])
    for idx, axx in enumerate(ax):
        axx[0].imshow(set[idx][0])
        axx[1].imshow(set[idx][1])
    # [x.axis('off') for x in ax.flatten()]
    plt.gcf().set_size_inches([2.4, 5])
    plt.suptitle(f'Size: f{set[0][0].shape}\n{extra_info}')

    [x.set_xticks([]) for x in ax.flatten()]
    [x.set_yticks([]) for x in ax.flatten()]

    plt.savefig(path)


def get_new_affine_values(transf_code):
    # Example transf_code = 's[0.2, 0.3]tr[0,90]'  -> s is within 0.2, 0.3, t is default, r is between 0 and 90 degrees
    def get_values(code):
        real_num = r'[-+]?[0-9]*\.?[0-9]+'
        try:
            return [float(i) for i in re.search(f'{code}\[({real_num}),\s?({real_num})]', transf_code).groups()]
        except AttributeError:
            if code == 't':
                return [-0.2, 0.2]
            if code == 's':
                return [0.7, 1.3]
            if code == 'r':
                return [0, 360]

    tr = [np.random.uniform(*get_values('t')), np.random.uniform(*get_values('t'))] if 't' in transf_code else (0, 0)
    scale = np.random.uniform(*get_values('s')) if 's' in transf_code else 1.0
    rot = np.random.uniform(*get_values('r')) if 'r' in transf_code else 0
    return {'rt': rot, 'tr': tr, 'sc': scale, 'sh': 0.0}


def my_affine(img, translate, **kwargs):
    return F.affine(img, translate=[int(translate[0] * img.size[0]), int(translate[1] * img.size[1])], **kwargs)


class RecordCossim(RecordActivations):
    def compute_cosine_pair(self, image0, image1):# path_save_fig, stats):
        cossim = {}

        self.net(make_cuda(image0.unsqueeze(0), torch.cuda.is_available()))
        first_image_act = {}
        activation_image1 = deepcopy(self.activation)
        for name, features1 in self.activation.items():
            if not np.any([i in name for i in self.only_save]):
                continue
            first_image_act[name] = features1.flatten()

        self.net(make_cuda(image1.unsqueeze(0), torch.cuda.is_available()))
        activation_image2 = deepcopy(self.activation)

        second_image_act = {}
        for name, features2 in self.activation.items():
            if not np.any([i in name for i in self.only_save]):
                continue
            second_image_act[name] = features2.flatten()
            if name not in cossim:
                cossim[name] = []
            cossim[name].append(torch.nn.CosineSimilarity(dim=0)(first_image_act[name], second_image_act[name]).item())

        return cossim

    def compute_random_set(self, image_folder, transform, fill_bk=None, affine_transf='', N=5, path_save_fig=None, base_name='base'):
        norm = [i for i in transform.transforms if isinstance(i, transforms.Normalize)][0]
        save_num_image_sets = 5
        levels = [os.path.basename(i) for i in glob.glob(image_folder + '/**')]
        if not levels:
            assert False
        sets = [np.unique([os.path.splitext(os.path.basename(i))[0] for i in glob.glob(image_folder + f'/{l}/*')]) for l in levels]
        assert np.all([len(sets[ix]) == len(sets[ix-1]) for ix in range(1, len(sets))]), "Length for one of the folder doesn't match other folder in the dataset"
        assert np.all([np.all(sets[ix] == sets[ix-1]) for ix in range(1, len(sets))]), "All names in all folders in the dataset needs to match. Some name didn't match"
        sets = sets[0]

        df = pd.DataFrame([])
        for a in tqdm(levels):
            save_sets = []
            for s in sets:

                save_fig = True
                for n in range(N):
                    im_0 = Image.open(image_folder + f'/{base_name}/{s}.png').convert('RGB')
                    im_i = Image.open(image_folder + f'/{a}/{s}.png').convert('RGB')
                    af = get_new_affine_values(affine_transf)
                    images = [my_affine(im, translate=af['tr'], angle=af['rt'], scale=af['sc'], shear=af['sh'], interpolation=InterpolationMode.NEAREST, fill=fill_bk) for im in [im_0, im_i]]

                    images = [transform(i) for i in images]
                    df_row = {'set': s, 'level': a, 'n': n}
                    cs = self.compute_cosine_pair(images[0], images[1]) #, path_fig='')
                    df_row.update(cs)
                    df = pd.concat([df, pd.DataFrame.from_dict(df_row)])

                    if save_fig:
                        save_sets.append([conver_tensor_to_plot(i, norm.mean, norm.std) for i in images])
                        if len(save_sets) == save_num_image_sets:
                            save_figs(path_save_fig + f'{s}_{a}.png', save_sets, extra_info=affine_transf + f'\nmeanCS[-1]: {np.mean(cs[self.all_layers_names[-1]]):.3f}')
                            save_fig = False
                            save_sets = []
                    plt.close('all')

        all_layers = list(cs.keys())
        return df, all_layers


def compute_cossim_from_img(config):
    config.model, norm_values, resize_value = GrabNet.get_net(config.network_name,
                                                              imagenet_pt=True if config.pretraining == 'ImageNet' else False)

    prepare_network(config.model, config, train=False)

    transf_list = [transforms.Resize(resize_value),
                   torchvision.transforms.ToTensor(),
                   torchvision.transforms.Normalize(norm_values['mean'], norm_values['std'])]

    transform = torchvision.transforms.Compose(transf_list)

    fill_bk = 'black' if config.background == 'black' or config.background == 'random' else config.background
    debug_image_path = config.result_folder + '/debug_img/'
    pathlib.Path(os.path.dirname(config.result_folder)).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.dirname(debug_image_path)).mkdir(parents=True, exist_ok=True)
    if not config.affine_transf_code and config.rep > 1:
        print(sty.fg.red + 'No affine transformation code, but repetition is > 1. Is this a mistake?' + sty.rs.fg)

    recorder = RecordCossim(net=config.model, use_cuda=False, only_save=config.save_layers)
    cossim_df , layers_names = recorder.compute_random_set(image_folder=config.image_folder,
                                            transform=transform,
                                            fill_bk=fill_bk,
                                            affine_transf=config.affine_transf_code,
                                            N=config.rep,
                                            path_save_fig=debug_image_path,
                                            base_name=config.base_name
                                            )


    save_path = config.result_folder + '/cossim.df'
    print(fg.red + f'Saved in {save_path}' + rs.fg)

    pickle.dump({'layers_names': layers_names, 'cossim_df': cossim_df}, open(save_path, 'wb'))
    return cossim_df, layers_names

if __name__ == '__main__':
    from src.utils.misc import ConfigSimple

    config = ConfigSimple(project_name='MindSet',
                          network_name='inception_v3',
                          pretraining='ImageNet',
                          image_folder='./data/NAPvsMP',
                          affine_transf_code='t[-0.2, 0.2]s[1,1.2]r',
                          result_folder=f'./results/NAPvsMP/',
                          background='black',
                          save_layers=['Conv2d', 'Linear'],  # to be saved, a layer must contain any of these words
                          rep=2,
                          base_name='base',
                          )

    cossim_df, layers_names = compute_cossim_from_img(config)
