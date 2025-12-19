'''
Make subject-level maps of energy pathways

Author: Moohebat
Date: 05/05/2025
'''

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import colormaps as cmaps
import seaborn as sns
from scipy.stats import zscore
from sklearn.decomposition import PCA
from scripts.utils import (geneset_expression, 
                           class_enrichment,
                           plot_schaefer_fsaverage)

# path to directories
path_data = './data/'
path_result = './results/'
path_fig = './figures/'

# load energy pathway gene sets
with open(path_result + 'energy_genelist_dict.pickle', 'rb') as f:
    energy_dict = pickle.load(f)

# load expression dictionary
with open(path_data + 'expression_dict_schaefer400.pickle', 'rb') as f:
    expression_schaefer400 = pickle.load(f)

with open(path_data + 'expression_ds01.pickle', 'rb') as f:
    expression_ds01 = pickle.load(f)

# make subj-level maps
# i will first filter each subj expression to have only ds>0.1 genes
for subj, exp in expression_schaefer400.items():
    expression_schaefer400[subj] = exp.loc[:, exp.columns.isin(expression_ds01.columns)]


pca = PCA(n_components=1)

energy_exp = {}
energy_mean = {}
energy_median = {}
energy_pc1 = {}

# getting expression matrices, pc1 and mean maps for energy pathways
for subj, exp in expression_schaefer400.items():

    # make nested dictionaries for each subject
    energy_exp[subj] = {}
    energy_mean[subj] = {}
    energy_median[subj] = {}
    energy_pc1[subj] = {}

    for pathway, genes in energy_dict.items():
        energy_exp[subj][pathway] = geneset_expression(exp, genes)
        energy_mean[subj][pathway] = np.mean(energy_exp[subj][pathway], axis=1)
        energy_median[subj][pathway] = np.median(energy_exp[subj][pathway], axis=1)
        energy_pc1[subj][pathway] = np.squeeze(pca.fit_transform(energy_exp[subj][pathway]))

# dict_keys(['9861', '10021', '12876', '14380', '15496', '15697'])

energy_main = ['glycolysis', 'ppp', 'tca', 'oxphos', 'lactate']

# plot subject level maps
for subj, exp in energy_mean.items():
    for pathway in energy_main:
        plot_schaefer_fsaverage(zscore(exp[pathway]), cmap=cmaps.matter_r)
        plt.title('subject' + subj + '-' + pathway)
        plt.tight_layout()
        # plt.savefig(path_fig+subj+'_'+pathway+'.svg', dpi=600)
        plt.show()


# pathway correlation among subjects
for pathway in energy_main:
    exp_df = pd.DataFrame({subj: energy_mean[subj][pathway] for subj in energy_mean.keys()})
    plt.figure(figsize=(4,4))
    mask = np.tril(np.ones(exp_df.shape[1]))
    ax = sns.heatmap(exp_df.corr(method='spearman'), annot=True, square=True,
                mask=mask,
                cmap=cmaps.BlueWhiteOrangeRed, cbar=True, linewidths=.4,
                linecolor='darkgrey',
                xticklabels=exp_df.columns,
                yticklabels=exp_df.columns,
                vmin=-1, vmax=1, alpha=1,
                cbar_kws={'shrink': 0.2, 'aspect': 10, 'ticks': [-1, 0, 1]},)
    plt.title('subj-subj corr - ' + pathway)
    plt.yticks(rotation=0)
    plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
    plt.tight_layout()
    # plt.savefig(path_fig+pathway+'_btw_subj_corr.svg')
    plt.show()


# class enrichment for each subject
# loading class labels
yeo_schaefer400 = np.load(path_data+'yeo_schaefer400.npy')
ve_schaefer400 = np.load(path_data+'ve_schaefer400.npy', allow_pickle=True)
mesulam_schaefer400 = np.load(path_data+'mesulam_schaefer400.npy', allow_pickle=True)

# loading spins
spins10k = np.load(path_data+'spins10k.npy')

#####################
# convert label names
# Mapping dictionary

# ve
mapping = {
    'association':'Ac',
    'association2':'Ac2',
    'insular': 'Ins',
    'limbic': 'Lim',
    'primary motor': 'PM',
    'primary sensory': 'PS',
    'primary/secondary sensory': 'PSS'
}
# Convert the values using the mapping dictionary
ve_schaefer400 = np.array([mapping[val] for val in ve_schaefer400])


classes = {
    've': {
        'order': ['PM','PS','PSS', 'Ac','Ac2','Ins','Lim'],
        'map': ve_schaefer400}}

# plot subject level netwrok class enrichment
# ve

ve_mean = {}
ve_pspin = {}
for subj, exp in energy_mean.items():
    energy_mean_df = pd.DataFrame(exp).reset_index(drop=True)
    energy_mean_df = energy_mean_df[energy_main]
    ve_mean[subj], ve_pspin[subj] = class_enrichment(energy_mean_df, 
                 ve_schaefer400, 
                 spins10k,
                 classes['ve']['order'],
                 path_fig,
                 subj+'ve_enrichment_new')
