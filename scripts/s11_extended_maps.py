'''
script running all analysis for
the extended energy pathway maps

Author: Moohebat
Date: 21/07/2024
'''
# brainspan analysis of extended maps is in the brainspan script.

#importing packages
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import colormaps as cmaps
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import spearmanr, zscore
from scripts.utils import corr_spin_test, pair_corr_spin, plot_schaefer_fsaverage
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update({'font.size': 8})

# path to directories
path_data = './data/'
path_result = './results/'
path_fig = './figures/'

# loading energy pathway data
with open(path_result+'energy_expression_matrix.pickle', 'rb') as f:
    energy_exp = pickle.load(f)
with open(path_result+'energy_mean_expression.pickle', 'rb') as f:
    energy_mean = pickle.load(f)
with open(path_result+'energy_pc1_expression.pickle', 'rb') as f:
    energy_pc1 = pickle.load(f)

# loading schaefer400 spins
spins1k = np.load(path_data+'spins1k.npy')

# plot all energy maps
# plotting mean gene expression
for key, value in energy_mean.items():
    plot_schaefer_fsaverage(zscore(value), cmap=cmaps.matter_r)
    plt.title(key)
    plt.savefig(path_fig+key+'_mean.svg', dpi=600)
    plt.show()

# cross correlation of all energy maps
# convert dict to dataframe
energy_mean_df = pd.DataFrame(energy_mean).reset_index(drop=True)
# drop redundant pathway maps
energy_df = energy_mean_df.drop(['etc', 'lactate_metabolism',
                                      'lactate_transport', 'kb_synth', 
                                      'kb_metabolism', 'fa_synth', 'betaox',
                                      'glycogen_synth', 'glycogen_cat',
                                      'pdp', 'pdk', 'pc'], axis=1)

################################
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

####################
#louvain clustering
import bct
from netneurotools import plotting, cluster
from netneurotools.modularity import consensus_modularity

corr = energy_df.corr('spearman')

def community_detection(A, gamma_range):
    nnodes = len(A)
    ngamma = len(gamma_range)
    consensus = np.zeros((nnodes, ngamma))
    qall = []
    zrand = []
    i = 0
    for g in gamma_range:
        consensus[:, i], q, z = consensus_modularity(A, g, B='negative_asym', 
                                                     repeats=1000)
        qall.append(q)
        zrand.append(z)
        i += 1
    return (consensus, qall, zrand)

gamma_range = [x/10.0 for x in range(1, 21, 1)]
consensus, qall, zrand = community_detection(corr.values, gamma_range)

for i in range(len(gamma_range)):
    plotting.plot_mod_heatmap(corr.values, np.squeeze(consensus[:, i]), 
                            vmin=-1, vmax=1,
                            xticklabels=corr.index,
                            yticklabels=corr.index,
                            mask_diagonal=False,
                            cmap=cmaps.BlueWhiteOrangeRed, 
                            square=True,)
    plt.title('gamma'+ str(gamma_range[i]))
    plt.tight_layout()
    plt.savefig(path_fig+'clustering_louvain_gamma'+str(gamma_range[i])+'.svg')
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
