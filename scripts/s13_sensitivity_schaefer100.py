
'''
Sensitivity robustness analysis
Using schaefer100 parcellation
Author: Moohebat
Date: 17/09/2024
'''

# get expression matrices for schaefer100
# plot mean maps and pc1 maps
# enrichment across cytoarchitectonic, intrinsic functional classes, labels from justine
# correlation heatmaps with other brain measures?

# import packages
import numpy as np
import pandas as pd
import pickle
import abagen
import seaborn as sns
import matplotlib.pyplot as plt
import colormaps as cmaps
from sklearn.decomposition import PCA
from scipy.stats import zscore, spearmanr
from nilearn.datasets import fetch_atlas_schaefer_2018
from scripts.utils import (plot_schaefer_fsaverage, corr_spin_test, 
                           load_expression, filter_expression_ds, 
                           geneset_expression)

# path to directories
path_data = './data/'
path_result = './results/'
path_fig = './figures/'
path_all= './results/energy_sets/all_pathways/'

# loading schaefer100 spins
spins1k = np.load(path_data+'spins1k_schaefer100.npy')
spins10k = np.load(path_data+'spins10k_schaefer100.npy')

# load expression data for schaefer100 parcellation
expression_schaefer100 = load_expression(scale=100)
# 100 regions x 15633 genes

# save left hemisphere expression dict for filtering
with open(path_data + 'expression_dict_schaefer100.pickle', 'wb') as f:
    pickle.dump(expression_schaefer100, f)
# load left hemisphere expression
with open(path_data + 'expression_dict_schaefer100.pickle', 'rb') as f:
    expression_schaefer100 = pickle.load(f)

# keeping genes with ds>0.1 and only the left hemisphere rows
expression100_ds01 = filter_expression_ds(expression_schaefer100, 
                                          ds=0.1)
# dataframe of 100 x 10832

# save
with open(path_data + 'expression100_ds01.pickle', 'wb') as f:
    pickle.dump(expression100_ds01, f)

# load 
with open(path_data + 'expression100_ds01.pickle', 'rb') as f:
    expression100_ds01 = pickle.load(f)

##################################
# load energy gene sets dictionary
with open(path_result + 'energy_genelist_dict.pickle', 'rb') as f:
    energy_dict = pickle.load(f)

# retrieving energy expression matrices
pca = PCA(n_components=1)

energy_exp_100 = {}
energy_mean_100 = {}
energy_pc1_100 = {}

# getting expression matrices, pc1 and mean maps for energy pathways
for key, value in energy_dict.items():
    energy_exp_100[key] = geneset_expression(expression100_ds01, value, key, path_result)
    energy_mean_100[key] = np.mean(energy_exp_100[key], axis=1)
    energy_pc1_100[key] = np.squeeze(pca.fit_transform(energy_exp_100[key]))

# saving
with open(path_result + 'energy_expression_matrix_100.pickle', 'wb') as f:
    pickle.dump(energy_exp_100, f)
with open(path_result + 'energy_mean_expression_100.pickle', 'wb') as f:
    pickle.dump(energy_mean_100, f)
with open(path_result + 'energy_pc1_expression_100.pickle', 'wb') as f:
    pickle.dump(energy_pc1_100, f)

# load
with open(path_result + 'energy_expression_matrix_100.pickle', 'rb') as f:
    energy_exp_100 = pickle.load(f)
with open(path_result + 'energy_mean_expression_100.pickle', 'rb') as f:
    energy_mean_100 = pickle.load(f)
with open(path_result + 'energy_pc1_expression_100.pickle', 'rb') as f:
    energy_pc1_100 = pickle.load(f)

# convert dict to datframe
energy_mean_df = pd.DataFrame.from_dict(energy_mean_100, 
                                        orient='columns').reset_index(drop=True)

#########################################
# comparison table for number of genes in gene list vs. ahba
size_table = pd.DataFrame(index=energy_dict.keys(),
                          columns=['gene_set', 'ahba'])
for pathway in energy_dict.keys():
    # size_table.loc[pathway, 'go'] = len(go_dict[pathway])
    # size_table.loc[pathway, 'reactome'] = len(reactome_dict[pathway])
    size_table.loc[pathway, 'gene_set'] = len(energy_dict[pathway])
    size_table.loc[pathway, 'ahba'] = energy_exp_100[pathway].shape[1]

size_table.to_csv(path_result+'comparison_table_100.csv')

##########
# analysis
# main energy pathways
main_energy = ['glycolysis', 'ppp', 'tca', 'oxphos', 'lactate']
energy_mean_df = energy_mean_df[main_energy]

# plotting mean gene expression
for key, value in energy_mean_100.items():
    if key in main_energy:
        plot_schaefer_fsaverage(zscore(value), cmap=cmaps.matter_r, resolution=100)
        plt.title(key)
        plt.savefig(path_fig+key+'_mean_100.svg')
        plt.show()

# plotting pc1 gene expression
for key, value in energy_pc1_100.items():
    if key in main_energy:
        plot_schaefer_fsaverage(value, cmap=cmaps.BlueWhiteOrangeRed, resolution=100)
        plt.title(key)
        plt.savefig(path_fig+key+'_pc1_100.svg')
        plt.show()


# correlation of energy maps
# plotting
plt.figure(figsize=(3.5,3.5))
sns.heatmap(energy_mean_df.corr('spearman'), annot=True, square=True,
            mask=np.tril(np.ones(energy_mean_df.shape[1])),
            cmap=cmaps.BlueWhiteOrangeRed, cbar=True, linewidths=.3,
            xticklabels=energy_mean_df.columns,
            yticklabels=energy_mean_df.columns,
            vmin=-1, vmax=1, alpha=1)
plt.yticks(rotation=0)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(path_fig+'pathway_mean_correlation_100.svg')
plt.show()

#################################
# gene co-expression and plotting

energy_exp_100 = {k: energy_exp_100[k] for k in main_energy}

exp_matrix_100 = pd.concat(energy_exp_100.values(), axis=1)
exp_matrix_100 = exp_matrix_100.loc[:,~exp_matrix_100.columns.duplicated()]
exp_matrix_100 = exp_matrix_100.reset_index(drop=True)

#sorted gene co-expression
i_glyco = np.array([list(exp_matrix_100.columns).index(i) for i in energy_exp_100['glycolysis'].columns])
i_ppp = np.array([list(exp_matrix_100.columns).index(i) for i in energy_exp_100['ppp'].columns])
i_tca = np.array([list(exp_matrix_100.columns).index(i) for i in energy_exp_100['tca'].columns])
i_oxphos = np.array([list(exp_matrix_100.columns).index(i) for i in energy_exp_100['oxphos'].columns])
i_lactate = np.array([list(exp_matrix_100.columns).index(i) for i in energy_exp_100['lactate'].columns])

classes = np.zeros((len(exp_matrix_100.columns), 1))
classes[i_glyco, 0] = 1
classes[i_ppp, 0] = 2
classes[i_tca, 0] = 3
classes[i_oxphos, 0] = 5
classes[i_lactate, 0] = 6

# plotting
import matplotlib.patches as patches
from netneurotools import plotting
class_names = ['glycolysis', 'ppp', 'tca', 'oxphos', 'lactate']
plt.rcParams.update({'font.size': 8})
fig, axs = plt.subplots(1, 1, figsize=(17,17))
axs = axs.ravel()
for i in range(1):
    inds = plotting.sort_communities(np.corrcoef(zscore(exp_matrix_100).T), classes[:, i])
    bounds = plotting._grid_communities(classes[:, i])
    sns.heatmap(data=np.corrcoef(zscore(exp_matrix_100).T)[np.ix_(inds, inds)],
                vmin=-1, vmax=1, ax=axs, cbar=True, square=True, cmap=cmaps.BlueWhiteOrangeRed,
                linewidths=0, xticklabels=exp_matrix_100.columns[inds], yticklabels=exp_matrix_100.columns[inds])
    for n, edge in enumerate(np.diff(bounds)):
        axs.add_patch(patches.Rectangle((bounds[n], bounds[n]), 
                                            edge, edge, fill=False,
                                            linewidth=4, edgecolor='black'))

plt.tight_layout()
plt.savefig(path_fig+'energy_coexpression_communit_sorted_100.svg')
plt.show()

################################################
# correlation of energy maps with pc1 expression
pca = PCA(n_components=1)
genepc1 = np.squeeze(pca.fit_transform(pd.concat(expression_schaefer100).groupby('label').mean()))

# calculate correlation and p_spins
r = []
pspin = []
for key, value in energy_mean_100.items():
        if key in main_energy:
            corr, _, p = corr_spin_test(genepc1, value, spins10k)
            r.append(corr)
            pspin.append(p)
        # plt.xlabel('ahba mean expression')
        # plt.ylabel(key+' mean expression')
        # plt.tight_layout()
        # plt.show()
        # plt.close()

plt.figure(figsize=(2.5,2))
bp = sns.barplot(x=r, y=main_energy, width=0.5, orient='h')
for i, patch in enumerate(bp.patches):
    if pspin[i] < 0.05:
        patch.set_facecolor('sandybrown')
    else:
        patch.set_facecolor('lightgrey')
# plt.xticks(rotation=45)
plt.xlim(-1,1)
plt.xlabel("Spearman's r")
plt.ylabel('pathway')
plt.title('corr with genepc1')
# plt.autoscale(enable=True, axis='both', tight=True)
plt.tight_layout()
sns.despine()
plt.savefig(path_fig+'energy_genepc1_corr_100.svg')
plt.show()
