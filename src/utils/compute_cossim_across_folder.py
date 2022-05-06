"""
Given a dataset folder `./data/xx/` and a base folder, it compares each sample in each folder with the base folder, e.g
    `./data/xx/base/0.png` vs `./data/xx/comp1/0.png`,
    `./data/xx/base/1.png` vs  `./data/xx/comp1/1.png`,
    .
    .
    .
    `./data/xx/base/0.png` vs `./data/xx/comp2/0.png`,
    .
    .
The number of samples in each folder must match.
Each comparison is done multiple time at different transformations
"""
from torchvision.transforms.functional import InterpolationMode
import glob
import pandas as pd
from src.utils.net_utils import GrabNet, prepare_network, make_cuda
from src.utils.misc import conver_tensor_to_plot
import matplotlib.pyplot as plt
import numpy as np
from sty import fg, bg, rs, ef
import pickle
import os
import pathlib
from tqdm import tqdm
import torchvision.transforms as transforms
import torchvision
import PIL.Image as Image
from src.utils.misc import get_new_affine_values, my_affine, save_figs
from src.utils.activation_recorder import RecordCossim

class RecordCossimAcrossFolders(RecordCossim):
    def compute_random_set(self, image_folder, transform, matching_transform=False, fill_bk=None, affine_transf='', N=5, path_save_fig=None, base_name='base'):
        norm = [i for i in transform.transforms if isinstance(i, transforms.Normalize)][0]
        save_num_image_sets = 5
        all_files = glob.glob(image_folder + '/**')
        levels = [os.path.basename(i) for i in glob.glob(image_folder + '/**')]

        sets = [np.unique([os.path.splitext(os.path.basename(i))[0] for i in glob.glob(image_folder + f'/{l}/*')]) for l in levels]
        assert np.all([len(sets[ix]) == len(sets[ix-1]) for ix in range(1, len(sets))]), "Length for one of the folder doesn't match other folder in the dataset"
        assert np.all([np.all(sets[ix] == sets[ix-1]) for ix in range(1, len(sets))]), "All names in all folders in the dataset needs to match. Some name didn't match"
        sets = sets[0]

        df = pd.DataFrame([])
        save_sets = []
        for s in tqdm(sets):
            plt.close('all')
            for a in levels:
                save_fig = True
                for n in range(N):
                    im_0 = Image.open(image_folder + f'/{base_name}/{s}.png').convert('RGB')
                    im_i = Image.open(image_folder + f'/{a}/{s}.png').convert('RGB')
                    af = [get_new_affine_values(affine_transf) for i in [im_0, im_i]] if not matching_transform else [get_new_affine_values(affine_transf)] * 2
                    images = [my_affine(im, translate=af[idx]['tr'], angle=af[idx]['rt'], scale=af[idx]['sc'], shear=af[idx]['sh'], interpolation=InterpolationMode.NEAREST, fill=fill_bk) for idx, im in enumerate([im_0, im_i])]

                    images = [transform(i) for i in images]
                    df_row = {'set': s, 'level': a, 'n': n}
                    cs = self.compute_cosine_pair(images[0], images[1]) #, path_fig='')
                    df_row.update(cs)
                    df = pd.concat([df, pd.DataFrame.from_dict(df_row)])

                    if save_fig:
                        save_sets.append([conver_tensor_to_plot(i, norm.mean, norm.std) for i in images])
                        if len(save_sets) == save_num_image_sets:
                            save_figs(path_save_fig + f'{s}_{a}', save_sets, extra_info=affine_transf)
                            save_fig = False
                            save_sets = []
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

    recorder = RecordCossimAcrossFolders(net=config.model, use_cuda=False, only_save=config.save_layers)
    cossim_df, layers_names = recorder.compute_random_set(image_folder=config.image_folder,
                                                           transform=transform,
                                                           matching_transform=config.matching_transform,
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
    from src.utils.misc import Config

    config = Config(project_name='MindSet',
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
