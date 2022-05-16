from torchvision.transforms.functional import InterpolationMode
import glob
import pandas as pd
from src.utils.net_utils import GrabNet, prepare_network
from src.utils.misc import conver_tensor_to_plot
import matplotlib.pyplot as plt
from sty import fg, rs
import pickle
import os
import pathlib
from tqdm import tqdm
import torchvision.transforms as transforms
import torchvision
import PIL.Image as Image
from src.utils.cosine_similarity.misc import get_new_affine_values, my_affine, save_figs
from src.utils.cosine_similarity.activation_recorder import RecordCossim

class RecordCossimImgBaseVsFolder(RecordCossim):
    def compute_random_set(self, folder, transform, matching_transform=False, fill_bk=None, affine_transf='', N=5, path_save_fig=None, base_image='base.png'):
        norm = [i for i in transform.transforms if isinstance(i, transforms.Normalize)][0]
        save_num_image_sets = 5
        compare_images = glob.glob(folder + '/**')


        df = pd.DataFrame([])
        save_sets = []
        for s in tqdm(compare_images):
            plt.close('all')
            save_fig = True
            for n in range(N):
                im_0 = Image.open(s).convert('RGB')
                im_i = Image.open(base_image).convert('RGB')
                af = [get_new_affine_values(affine_transf) for i in [im_0, im_i]] if not matching_transform else [get_new_affine_values(affine_transf)]*2
                images = [my_affine(im, translate=af[idx]['tr'], angle=af[idx]['rt'], scale=af[idx]['sc'], shear=af[idx]['sh'], interpolation=InterpolationMode.NEAREST, fill=fill_bk) for idx, im in enumerate([im_0, im_i])]

                images = [transform(i) for i in images]
                df_row = {'compare_img': os.path.basename(s), 'n': n}
                cs = self.compute_cosine_pair(images[0], images[1]) #, path_fig='')
                df_row.update(cs)
                df = pd.concat([df, pd.DataFrame.from_dict(df_row)])

                if save_fig:
                    save_sets.append([conver_tensor_to_plot(i, norm.mean, norm.std) for i in images])
                    if len(save_sets) == save_num_image_sets:
                        save_figs(path_save_fig + f'{os.path.basename(s)}', save_sets, extra_info=affine_transf)
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

    recorder = RecordCossimImgBaseVsFolder(net=config.model, use_cuda=True, only_save=config.save_layers)
    cossim_df, layers_names = recorder.compute_random_set(folder=config.folder,
                                                           transform=transform,
                                                           matching_transform=config.matching_transform,
                                                           fill_bk=fill_bk,
                                                           affine_transf=config.affine_transf_code,
                                                           N=config.rep,
                                                           path_save_fig=debug_image_path,
                                                           base_image=config.base_image
                                                           )


    save_path = config.result_folder + '/cossim.df'
    print(fg.red + f'Saved in {save_path}' + rs.fg)

    pickle.dump({'layers_names': layers_names, 'cossim_df': cossim_df, 'folder': config.folder, 'base_image': config.base_image}, open(save_path, 'wb'))
    return cossim_df, layers_names

if __name__ == '__main__':
    from src.utils.misc import ConfigSimple

    config = ConfigSimple(project_name='MindSet',
                          network_name='inception_v3',
                          pretraining='ImageNet',
                          affine_transf_code='t[-0.2, 0.2]s[0.5,0.9]r',
                          result_folder=f'./results/closure/square/full_vs_segm/',
                          background='black',
                          matching_transform=True,
                          save_layers=['Conv2d', 'Linear'],  # to be saved, a layer must contain any of these words
                          rep=2,
                          base_image='./data/closure/square/normal_full/0.png',
                          folder='./data/closure/square/angles_rnd/',
                          )

    cossim_df, layers_names = compute_cossim_from_img(config)
