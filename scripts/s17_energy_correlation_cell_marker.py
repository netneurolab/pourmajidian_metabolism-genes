
'''
Compare energy maps to individual cell type markers
for inhibitory and excitatory subtypes

Author: moohebat
'''

import numpy as np
import pandas as pd
import pickle
import abagen
import matplotlib.pyplot as plt
import colormaps as cmaps
from sklearn.decomposition import PCA
from scripts.utils import (pair_corr_spin, plot_heatmap, 
                           geneset_expression, plot_schaefer_fsaverage)
from statsmodels.stats.multitest import multipletests

# path to directories
path_data = './data/'
path_result = './results/'
path_fig = './figures/'

# schaefer 400
with open(path_data + 'expression_ds01.pickle', 'rb') as f:
    expression_ds01 = pickle.load(f)

# loading energy pathway data
with open(path_result+'energy_mean_expression.pickle', 'rb') as f:
    energy_mean = pickle.load(f)

# keep only main energy pathways
energy_mean_df = pd.DataFrame(energy_mean)
keep = ['glycolysis', 'ppp', 'tca', 'oxphos', 'lactate']
energy_mean_df = energy_mean_df[keep]

# loading schaefer400 spins
spins10k = np.load(path_data+'spins10k.npy')

# gene sets from kang
cell_genesets = {'pan_gaba': ['GAD1', 'GAD2'],
                 'parvalbumin': ['PVALB'],
                 'somatostatin': ['SST'],
                 'calbindin': ['CALB1', 'CALB2'],
                 'vip': ['VIP'],
                 'l1_exc': ['RELN'],
                 'l24_exc': ['CUX1', 'UNC5D'],
                 'l4_exc': ['RORB'],
                 'l5_exc': ['FEZF2', 'BCL11B', 'OTX1', 'ETV1'],
                 'l6_exc': ['ZFPM2', 'NTSR1', 'TLE4', 'FOXP2', 'TBR1', 'SOX5', 'SSTR2'],
                 'betz': ['ASGR2', 'CSN1S1'],
                 }

# make gene expression maps
geneset_exp = {}
geneset_mean = {}

for key, value in cell_genesets.items():
    geneset_exp[key] = geneset_expression(expression_ds01, value, key, path_result)
    geneset_mean[key] = np.mean(geneset_exp[key], axis=1)


# make latex table of cell markers
cell_markers = {}
for key, value in geneset_exp.items():
    cell_markers[key] = list(value.columns)

cell_marker_df = pd.Series(cell_markers).apply(", ".join).reset_index()
cell_marker_df.columns = ["cell", "marker"]

cell_marker_df.style.to_latex(path_result+'cell_marker_df.tex',
                              caption="", 
                              label="tab:cell_marker_df",)

# analysis
# convert to dataframe
geneset_mean_df = pd.DataFrame(geneset_mean).reset_index(drop=True)

# plot
for cell_type in geneset_mean_df.columns:
    plot_schaefer_fsaverage(geneset_mean_df[cell_type], cmap=cmaps.matter_r)
    plt.title(cell_type)
    plt.tight_layout()
    plt.savefig(path_fig+cell_type+'_exp_markers.svg')
    # plt.show()

# correlation with energy maps
corrs, pspins = pair_corr_spin(energy_mean_df, 
                                geneset_mean_df,
                                spins10k)

'''
corrs
              glycolysis       ppp       tca    oxphos   lactate
pan_gaba       -0.170791  0.411691  0.149261 -0.088142  0.046254
parvalbumin     0.237652  0.724668  0.621481  0.281438  0.654780
somatostatin    0.172928 -0.670812 -0.353746  0.194430 -0.215145
calbindin      -0.268501 -0.563358 -0.547430 -0.280768 -0.648872
vip            -0.255009 -0.208353 -0.332894 -0.212062 -0.324056
l1_exc         -0.251232 -0.414115 -0.506941 -0.285675 -0.430657
l24_exc        -0.364092  0.476572 -0.066644 -0.363980 -0.041773
l4_exc         -0.217602  0.574332  0.182873 -0.253502  0.097403
l5_exc          0.572327 -0.065207  0.343523  0.597566  0.472418
l6_exc          0.147969 -0.219318 -0.168586  0.076644 -0.000471
betz            0.630828  0.087947  0.494961  0.597362  0.689136

pspins
              glycolysis       ppp       tca    oxphos   lactate
pan_gaba        0.432457  0.000400  0.482552  0.676432  0.798120
parvalbumin     0.532047  0.000100  0.012899  0.432557  0.001700
somatostatin    0.609939  0.000200  0.417258  0.563944  0.660334
calbindin       0.396360  0.000700  0.025097  0.344166  0.001300
vip             0.037996  0.121188  0.004100  0.109089  0.008999
l1_exc          0.291671  0.007899  0.003100  0.179382  0.023798
l24_exc         0.053895  0.000400  0.793621  0.054895  0.827317
l4_exc          0.422958  0.002000  0.719128  0.344566  0.865913
l5_exc          0.002800  0.784722  0.222578  0.001600  0.027397
l6_exc          0.378662  0.140486  0.449355  0.604940  0.999200
betz            0.000300  0.793221  0.100990  0.001100  0.000400

'''

# fdr correction for multiple testing
model_pval = multipletests(pspins.values.flatten(), method='fdr_bh')[1]
model_pval = pd.DataFrame(model_pval.reshape(16,5))
model_pval.columns = pspins.columns
model_pval.index  = pspins.index

'''
model_pval
              glycolysis       ppp       tca    oxphos   lactate
pan_gaba        0.610016  0.003666  0.647326  0.791570  0.844166
parvalbumin     0.696728  0.003666  0.039413  0.610016  0.008499
somatostatin    0.745481  0.003666  0.610016  0.721323  0.789530
calbindin       0.610016  0.005499  0.069018  0.574276  0.007944
vip             0.094991  0.246864  0.015032  0.230765  0.029115
l1_exc          0.517481  0.027154  0.012177  0.340207  0.068888
l24_exc         0.125800  0.003666  0.844166  0.125800  0.858537
l4_exc          0.610016  0.009166  0.824001  0.574276  0.881949
l5_exc          0.011845  0.844166  0.408059  0.008499  0.071755
l6_exc          0.610016  0.275955  0.617863  0.745481  0.999200
betz            0.003666  0.844166  0.222178  0.007562  0.003666

'''

# plot
plt.figure(figsize=(5,3))
plot_heatmap(corrs, model_pval)
plt.savefig(path_fig+'cell_types__markers_energy_corr_heatmap.svg')
plt.show()
