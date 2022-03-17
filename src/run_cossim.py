from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from src.utils.misc import Config
from src.utils.compute_cossim import compute_cossim_from_img


def run_cossim(network_name, pretraining, image_folder, affine_transf_code):
    config = Config(project_name='MindSet',
                    network_name=network_name,
                    pretraining=pretraining,  # get_model_path(config, resume=True)
                    image_folder=image_folder[0],
                    affine_transf_code=affine_transf_code,
                    exp_folder=f'./results//{image_folder[1]}/',
                    # is_pycharm = True if 'PYCHARM_HOSTED' in os.environ else False,
                    )


    compute_cossim_from_img(config)

if __name__ == '__main__':
    pretraining = ['ImageNet']
    network_names = ['alexnet', 'densenet201', 'vgg19bn', 'resnet152', 'vonenet-resnet50', 'cornet-s', 'vonenet-cornets']
    # background = ['random', 'black', 'white']
    folders = (('./data/brackets', 'brackets'), ('./data/linearity/', 'lin'))
    transf_code = ['t[-0.2,0.4]', 'none', 's', 'r']  #, 'ts'] none']
    all_exps = (product(pretraining, network_names, folders, transf_code))
    arguments = list((dict(pretraining=i[0], network_name=i[1], image_folder=i[2], affine_transf_code=i[3]) for i in all_exps))
    [run_cossim(**a) for a in arguments]
    plt.close('all')
