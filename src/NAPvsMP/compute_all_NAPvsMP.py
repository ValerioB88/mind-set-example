"""
This will compute the folder-to-folder version of cosine similarity.
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
import numpy as np
import matplotlib.pyplot as plt
from src.utils.misc import ConfigSimple
import pickle
import sty
from src.cosine_similarity.compute_cossim import compute_cossim_from_img

def run(base_name, dataset_name):
    color_cycle = np.array(plt.rcParams['axes.prop_cycle'].by_key()['color'])

    config = ConfigSimple(network_name='inception_v3',
                          pretraining='ImageNet',
                          image_folder=f'./data/{dataset_name}',
                          affine_transf_code='',
                          background='black',
                          result_folder=f'./results/{dataset_name}_{base_name}',
                          save_layers=['Conv2d', 'Linear'],  # ReLU, MaxPool..
                          rep=10,
                          base_name=base_name
                          )
    print(sty.fg.red + f"Folder: {dataset_name}" + sty.rs.fg)
    cossim_df, layers_names = compute_cossim_from_img(config)

    ## Sample Analyises, you can just delete all of this if you want
    pk = pickle.load(open(config.result_folder + '/cossim.df', 'rb'))
    cossim_df, layers_names = pk['cossim_df'], pk['layers_names']
    cossim_ll = cossim_df[['level'] + layers_names]
    m = cossim_ll.groupby('level').mean()
    std = cossim_ll.groupby('level').std()
    len(cossim_df['set'].unique())

    ## Plot All Together
    import seaborn as sns
    plt.close('all')
    sns.set(style='white')
    levels = m.index.get_level_values('level')
    for idx, l in enumerate(levels):
        plt.plot(m.loc[l],  color=color_cycle[idx], marker='o', label=l)
        plt.fill_between(range(len(m.loc[l])), m.loc[l] - std.loc[l][idx], m.loc[l] + std.loc[l][idx], alpha=0.2, color=color_cycle[idx])

    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.savefig(config.result_folder + '/plot_layers.png')

    ## Perform One Way ANOVA
    import scipy.stats as stats
    r = stats.f_oneway(*[cossim_df[layers_names[-1]][cossim_df['level'] == i] for i in cossim_df['level'].unique()])
    with open(config.result_folder + '/stats_output.txt', 'w') as o:
        o.write(f'One Way Anova: p-value={r.pvalue}\n')

    ## Perform the repeated measures ANOVA
    try:
        from statsmodels.stats.anova import AnovaRM
        r = AnovaRM(data=cossim_df, depvar=layers_names[-1], subject='set', within=['level'], aggregate_func='mean').fit()
        with open(config.result_folder + '/stats_output.txt', 'a') as o:
            o.write("Repeated Measures ANOVA:\n")
            o.write(str(r.anova_table))
        print(r.summary())
    except:
        pass

    ## Barplot
    plt.close('all')
    m = cossim_df.groupby(['level', 'set'])[layers_names[-1]].mean()
    s = cossim_df.groupby(['level', 'set'])[layers_names[-1]].std()
    levels = m.index.get_level_values('level').unique()
    color_cycle = np.array(plt.rcParams['axes.prop_cycle'].by_key()['color'])

    for idx, l in enumerate(levels):
        span = 0.7
        width = span / (len(levels))
        i = np.arange(0, len(m.loc[l]))
        rect = plt.bar(np.arange(len(m.loc[l])) - span / 2
                       + width/2 + width * idx, m.loc[l], width, yerr=s.loc[l], label=l, color=color_cycle[idx], alpha=0.6)

    plt.ylabel('cosine similarity')
    plt.xticks(range(len(m.loc[l])), m.loc[l].index)
    plt.title(f'Comparing {base_name} with...')
    plt.legend()
    plt.savefig(config.result_folder + '/barplot.png')

    ##

fd = 'NAPvsMP'
run('base', f'{fd}/NAPvsMP_standard')
run('NS', f'{fd}/NAPvsMPlines')
run('base', f'{fd}/NAPvsMPnoshades')
run('base', f'{fd}/NAPvsMPsilh')



