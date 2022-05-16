"""
This will compute the base image vs folder cosine similarity.
Given a dataset folder `./data/xx/` and a base IMAGE, it compares the base image for each image in the target folder
    `./data/xx/base/0.png` vs `./data/xx/comp1/0.png`,
    `./data/xx/base/0.png` vs  `./data/xx/comp1/1.png`,
    .
    .
    .
    `./data/xx/base/0.png` vs `./data/xx/comp2/0.png`,
    .
    .
Each comparison is done multiple time at different transformations
"""

from src.utils.misc import ConfigSimple
from src.cosine_similarity_method.utils.compute_cossim_base_vs_folder import compute_cossim_from_img
import os

def run(base_image_folder, folder):
    base_image = f'{base_image_folder}/0.png'
    config = ConfigSimple(network_name='inception_v3',
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
    ### Do your analysis ##
    # . . .


folder_struct = 'closure/square/segm15/'

run(f'./data/{folder_struct}/normal_full/0.png', f'./data/{folder_struct}/normal_segmented')

