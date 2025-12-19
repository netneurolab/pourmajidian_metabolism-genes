
'''
script running all analysis for
the extended energy pathway maps

1. Plot brain maps in schaefer-400
2. Cross-correlation matrix
3. Clustering

Author: Moohebat
Date: 21/07/2024
'''
# for brainspan analysis of extended maps see s8_brainspan_rna.py

#importing packages
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import colormaps as cmaps
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import zscore
from statsmodels.stats.multitest import multipletests
from scripts.utils import plot_heatmap, pair_corr_spin, plot_schaefer_fsaverage

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update({'font.size': 8})

# path to directories
path_data = './data/'
path_result = './results/'
path_fig = './figures/'

# load energy pathway data
with open(path_result+'energy_expression_matrix.pickle', 'rb') as f:
    energy_exp = pickle.load(f)
with open(path_result+'energy_mean_expression.pickle', 'rb') as f:
    energy_mean = pickle.load(f)
with open(path_result+'energy_pc1_expression.pickle', 'rb') as f:
    energy_pc1 = pickle.load(f)

# load schaefer400 spins
spins10k = np.load(path_data+'spins10k.npy')

# plot mean gene expression
for key, value in energy_mean.items():
    plot_schaefer_fsaverage(zscore(value), cmap=cmaps.matter_r)
    plt.title(key)
    plt.savefig(path_fig+key+'_mean.svg', dpi=600)
    plt.show()


# cross correlation of all energy maps
# convert dict to dataframe
energy_mean_df = pd.DataFrame(energy_mean).reset_index(drop=True)

# keep non-redundant maps
extended_maps = ['glycolysis', 'ppp', 'tca', 'oxphos', 'lactate',
                 'complex1', 'complex2', 'complex3', 'complex4','atpsynth', 
                 'kb_util', 'fa_metabolism', 'glycogen_metabolism', 'bcaa_cat',
                 'pdc', 'mas', 'gps', 'creatine_kinase', 'ros_detox', 'ros_gen',
                 'no_signalling', 'atpase', 'gln_glu_cycle']

energy_df = energy_mean_df[extended_maps]



# check if maps show clustering
pca = PCA(n_components=2)
pca_exp = pca.fit_transform(zscore(energy_df.T))

pc_df = pd.DataFrame(data=pca_exp, columns=['PC1', 'PC2'])
pc_df['pathway'] = energy_df.T.index

# plot colored by pathway
plt.figure(figsize=(5,4))
sns.scatterplot(data=pc_df, x='PC1', y='PC2', 
                hue='pathway', s=57, linewidth=0, palette='tab20',
                legend=False)

for line in range(0, pc_df.shape[0]):
    plt.text(pc_df.PC1[line]+0.7, pc_df.PC2[line]+0.2,
            pc_df.pathway[line], horizontalalignment='left',
            size='small', color='black')
plt.title('PCA on energy pathway maps')
sns.despine()
plt.tight_layout()
plt.savefig(path_fig+'energy_pca_clustering.svg')
plt.show()


##################################
# extended maps correlation matrix

# simple
correlations = energy_df.corr('spearman')
plt.figure(figsize=(7,7))
sns.heatmap(correlations,
                cmap=cmaps.BlueWhiteOrangeRed,
                cbar=True,
                square=True,
                vmin=-1, vmax=1,
                linewidths=0.3,
                linecolor='lightgrey',
                cbar_kws={'shrink': 0.4, 'aspect':10, 'ticks': [-1, 0, 1]},
                )

plt.tight_layout()
plt.savefig(path_fig+'energy_extended_corrs.svg')
plt.show()

# with spin
extended_corrs, extended_pspins = pair_corr_spin(energy_df, energy_df, spins10k)

extended_corrs.to_csv(path_result+'extended_corrs.csv')
extended_pspins.to_csv(path_result+'extended_pspins.csv')

# fdr correction for multiple testing
model_pval = multipletests(extended_pspins.values.flatten(), method='fdr_bh')[1]
model_pval = pd.DataFrame(model_pval.reshape(23,23))
model_pval.columns = extended_pspins.columns
model_pval.index  = extended_pspins.index

model_pval.to_csv(path_result+'extended_pspins_fdr_pval.csv')

# plot
plot_heatmap(extended_corrs, model_pval, asteriks=False,
             edge=True,
             square=True,)
plt.savefig(path_fig+'extended_maps_corr_heatmap_pspin.svg')
plt.show()


# sns hierarchical clustering
# extended nergy pathways clustermap, 400-region wise
g = sns.clustermap(energy_df.T,
                    cmap=cmaps.matter_r,
                    cbar=True,
                    col_cluster=False,
                    figsize=(5, 7),)
plt.tight_layout()
plt.savefig(path_fig+'energy_pathway_clustermap_regions.svg')
plt.show()


##########################
# mitochondrial complexes
# correlation heatmap
mt_complexes = energy_mean_df[['complex1', 'complex2', 
                               'complex3', 'complex4',
                               'atpsynth']]

# mean correlation
mt_corr = np.unique(np.triu(mt_complexes.corr('spearman'), k=1))
mt_corr_mean = np.mean(mt_corr[mt_corr>0]) # 0.687

mt_corrs, pspins = pair_corr_spin(mt_complexes, mt_complexes, spins1k)
mask = np.tril(np.ones(mt_complexes.shape[1]))
plt.figure(figsize=(4,4))
ax = sns.heatmap(mt_corrs, annot=True, square=True,
            mask=mask,
            cmap=cmaps.BlueWhiteOrangeRed, cbar=True, linewidths=.3,
            xticklabels=mt_complexes.columns,
            yticklabels=mt_complexes.columns,
            vmin=-1, vmax=1, alpha=1)

# to add significance level
for i in range(pspins.shape[0]):
    for j in range(pspins.shape[1]):
        if mask[i, j] and i != j:
            if pspins.iloc[i, j] < 0.05:
                ax.text(i+0.6, j+0.4, '*', ha='left', va='bottom', color='k')
plt.yticks(rotation=0)
plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
plt.tight_layout()
plt.savefig(path_fig+'mito_complex_corrs.svg')
plt.show()
