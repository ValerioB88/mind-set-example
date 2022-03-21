import numpy as np
import matplotlib.pyplot as plt
from src.utils.misc import Config
import pickle
import sty
import pandas as pd
from src.utils.compute_cossim import compute_cossim_from_img
# name_data = 'NAPvsMPsilh'
# name_data = 'NAPvsMPnoshades'
# name_data = 'NAPvsMP'
name_data = 'NAPvsMPlines'

base_name = 'S'
config = Config(project_name='MindSet',
                network_name='inception_v3',
                pretraining='ImageNet',
                image_folder=f'./data/{name_data}',
                affine_transf_code='t[-0.2, 0.2]s[1,1.2]r',
                result_folder=f'./results/{name_data}_{base_name}',
                background='black',
                save_layers=['Conv2d', 'Linear'],  # to be saved, a layer must contain any of these words
                rep=50,
                base_name=base_name
                )
print(sty.fg.red + f"Folder: {name_data}" + sty.rs.fg)
cossim_df, layers_names = compute_cossim_from_img(config)


## Quick Analyiss
pk = pickle.load(open(config.result_folder + '/cossim.df', 'rb'))
cossim_df, layers_names = pk['cossim_df'], pk['layers_names']
cossim_ll = cossim_df[['alternative'] + layers_names]
m = cossim_ll.groupby('alternative').mean()
std = cossim_ll.groupby('alternative').std()
len(cossim_df['set'].unique())

## And Plot All Together
import seaborn as sns
plt.close('all')
sns.set(style='white')
color_cycle = np.array(plt.rcParams['axes.prop_cycle'].by_key()['color'])
levels = m.index.get_level_values('alternative')
for idx, l in enumerate(levels):
    plt.plot(m.loc[l],  color=color_cycle[idx], marker='o', label=l)
    plt.fill_between(range(len(m.loc[l])), m.loc[l] - std.loc[l][idx], m.loc[l] + std.loc[l][idx], alpha=0.2, color=color_cycle[idx])

plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.savefig(config.result_folder + '/plot_layers.png')

##
# Perform the repeated measures ANOVA
from statsmodels.stats.anova import AnovaRM
r = AnovaRM(data=cossim_df, depvar=layers_names[-1], subject='set', within=['alternative'], aggregate_func='mean').fit()
with open(config.result_folder + '/stats_output.txt', 'w') as o:
    o.write(str(r.anova_table))
print(r.summary())

##
plt.close('all')
m = cossim_df.groupby(['alternative', 'set'])[layers_names[-1]].mean()
s = cossim_df.groupby(['alternative', 'set'])[layers_names[-1]].std()
levels = m.index.get_level_values('alternative').unique()

for idx, l in enumerate(levels):
    plt.bar(range(len(m.loc[l])), m.loc[l], yerr=s.loc[l], label=l, alpha=0.5)
plt.ylabel('cosine similarity')
plt.xticks(range(len(m.loc[l])), m.loc[l].index)
plt.legend()
plt.savefig(config.result_folder + '/plot_NAPvsMP.png')

##


##

