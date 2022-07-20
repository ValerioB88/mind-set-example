
import numpy as np
import matplotlib.pyplot as plt
from src.utils.misc import ConfigSimple
import pickle
import sty
import argparse
import os
color_cycle = np.array(plt.rcParams['axes.prop_cycle'].by_key()['color']*3)


parser = argparse.ArgumentParser()
parser.add_argument('--pickle_path')
parser.add_argument('--result_folder', default=None)

args = parser.parse_known_args()[0]

# args = parser.parse_args(['--pickle_path', './results/NAPvsMP/NAPvsMPlines/cossim_df.pickle'
# ])

if args.result_folder is None:
    args.result_folder = os.path.dirname(args.pickle_path)
print(sty.fg.red + "Performing standard cosine similarity analysis for " +sty.fg.green + f"{args.pickle_path}" + sty.rs.fg)

## Sample Analyises, you can just delete all of this if you want
pk = pickle.load(open(args.pickle_path, 'rb'))
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
plt.savefig(args.result_folder + '/plot_layers.png')

## Perform One Way ANOVA
import scipy.stats as stats
r = stats.f_oneway(*[cossim_df[layers_names[-1]][cossim_df['level'] == i] for i in cossim_df['level'].unique()])
with open(args.result_folder + '/stats_output.txt', 'w') as o:
    o.write(f'One Way Anova: p-value={r.pvalue}\n')

## Perform the repeated measures ANOVA
try:
    from statsmodels.stats.anova import AnovaRM
    r = AnovaRM(data=cossim_df, depvar=layers_names[-1], subject='set', within=['level'], aggregate_func='mean').fit()
    with open(args.result_folder + '/stats_output.txt', 'a') as o:
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
plt.subplots(1, 1, figsize=(18, 5))
for idx, l in enumerate(levels):
    span = 0.7
    width = span / (len(levels))
    i = np.arange(0, len(m.loc[l]))
    rect = plt.bar(np.arange(len(m.loc[l])) - span / 2
                   + width/2 + width * idx, m.loc[l], width, yerr=s.loc[l], label=l, color=color_cycle[idx], alpha=0.6)

plt.ylabel('cosine similarity')
plt.xticks(range(len(m.loc[l])), m.loc[l].index, rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig(args.result_folder + '/barplot.png')

print(sty.fg.red + f"Analysis finished, everything saved in " + sty.fg.green + f"{args.result_folder}" + sty.rs.fg)
##

