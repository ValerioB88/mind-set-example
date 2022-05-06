import numpy as np
import matplotlib.pyplot as plt
from src.utils.misc import Config
import pickle
import sty
import pandas as pd
from src.utils.compute_cossim_base_vs_folder import compute_cossim_from_img
import os
def run(base_image_folder, folder):
    base_image = f'{base_image_folder}/0.png'
    config = Config(network_name='inception_v3',
                    pretraining='ImageNet',
                    folder=folder,
                    affine_transf_code='r',
                    background='black',
                    matching_transform=True,
                    result_folder=f'./results/{folder_struct}/' + ('same_transf' if True else 'diff_transf') + f'/{os.path.basename(base_image_folder)}_vs_{os.path.basename(folder)}',
                    save_layers=['Conv2d', 'Linear'],  # ReLU, MaxPool..
                    rep=10,
                    base_image=base_image
                    )
    cossim_df, layers_names = compute_cossim_from_img(config)

folder_struct = 'closure/square/segm15/'

run(f'./data/{folder_struct}/normal_full/0.png', f'./data/{folder_struct}/normal_segmented')

