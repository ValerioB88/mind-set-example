"""
This will compute the base image vs folder cosine similarity.
Given a dataset folder `./data/xx/` and a base IMAGE, it compares the base image for each image in the target folder
    `./data/xx/base/0.png` vs `./data/xx/comp1/0.png`,
    `./data/xx/base/0.png` vs  `./data/xx/comp1/1.png`,
    .
    .
    .
    .
Each comparison is done multiple time at different transformations
"""

from src.utils.misc import ConfigSimple
from src.utils.cosine_similarity.compute_cossim_base_vs_folder import compute_cossim_from_img


def run(base_image, folder, result_folder):
    config = ConfigSimple(network_name='inception_v3',
                          pretraining='ImageNet',
                          folder=folder,
                          affine_transf_code='r',
                          background='black',
                          matching_transform=True,
                          result_folder=result_folder,
                          save_layers=['Conv2d', 'Linear'],  # ReLU, MaxPool..
                          rep=10,
                          base_image=base_image
                          )
    cossim_df, layers_names = compute_cossim_from_img(config)
    ### Do your analysis ##
    # . . .


folder_struct = 'closure/square/segm15/'

run(base_image=f'./data/{folder_struct}/normal_full/0.png',
    folder=f'./data/{folder_struct}/angles_rnd',
    result_folder=f'./results/{folder_struct}/normal_full_0_vs_angles_rnd_folder.pickle')

