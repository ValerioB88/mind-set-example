from src.utils.misc import Config
import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
plt.close('all')
sns.set(style='white')
all_sizes = np.arange(-0.05, 0.05, 0.005)


dataset_name = 'ebbinghaus'
base_name = 'large'
def get_config(base_name):
    config = Config(network_name='inception_v3',
                    pretraining='ImageNet',
                    affine_transf_code='',
                    background='black',
                    result_folder=f'./results/{dataset_name}_{base_name}',
                    base_name=base_name
                    )
    return config

config = get_config('small')
pk = pickle.load(open(config.result_folder + '/cossim.df', 'rb'))
cossim_df, layers_names = pk['cossim_df'], pk['layers_names']
cossim_ll = cossim_df[['level'] + layers_names]
m = cossim_ll.groupby('level').mean()
std = cossim_ll.groupby('level').std()
len(cossim_df['set'].unique())

## And Plot All Together
levels = list(m.index.get_level_values('level'))
levels.remove('small' if base_name == 'large' else 'large')
plt.plot(m[layers_names[-1]].reindex([f'{s:.3f}' for s in all_sizes]), label='small')

##
config = get_config('large')
pk = pickle.load(open(config.result_folder + '/cossim.df', 'rb'))
cossim_df, layers_names = pk['cossim_df'], pk['layers_names']
cossim_ll = cossim_df[['level'] + layers_names]
m = cossim_ll.groupby('level').mean()
std = cossim_ll.groupby('level').std()
len(cossim_df['set'].unique())

## And Plot All Together
levels = list(m.index.get_level_values('level'))
levels.remove('small' if base_name == 'large' else 'large')
plt.plot(m[layers_names[-1]].reindex([f'{s:.3f}' for s in all_sizes]), label='large')

plt.legend()
plt.xlabel('Size small circle')
plt.ylabel('Cos. Sim.')