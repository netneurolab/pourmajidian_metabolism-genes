
'''
Sensitivity analysis
Using only left hemisphere data
Author: Moohebat
Date: 09/07/2024
'''

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
                           filter_expression_ds, geneset_expression)

# path to directories
path_data = './data/'
path_result = './results/'
path_fig = './figures/'
path_all= './results/energy_sets/all_pathways/'

# loading schaefer400 spins
# spins are generated for one hemisphere and the mirrored
# so I can just take the first 200 elements for lh
spins10k = np.load(path_data+'spins10k.npy')

# function to get expression data from only the left hemisphere
# setting lr_mirror to None
def load_expression_lh(scale):

    schaefer = fetch_atlas_schaefer_2018(n_rois=scale) # 7networks and 1mm res.

    expression = abagen.get_expression_data(schaefer['maps'],
                                            lr_mirror=None, 
                                            missing='interpolate', 
                                            return_donors=True)
    return expression

# load expression data for schaefer400 parcellation
lh_expression_schaefer400 = load_expression_lh(scale=400)

# save left hemisphere expression dict for filtering
with open(path_data + 'lh_expression_dict_schaefer400.pickle', 'wb') as f:
    pickle.dump(lh_expression_schaefer400, f)
# load left hemisphere expression
with open(path_data + 'lh_expression_dict_schaefer400.pickle', 'rb') as f:
    lh_expression_schaefer400 = pickle.load(f)

# keeping genes with ds>0.1
lh_expression_ds01 = filter_expression_ds(lh_expression_schaefer400, 
                                          ds=0.1)
# dataframe of 400 x 8376

# save
path_data = 'D:/McGill/Dagher_lab/my_neuro_project/project_data/'
with open(path_data + 'lh_expression_ds01.pickle', 'wb') as f:
    pickle.dump(lh_expression_ds01, f)

# load
with open(path_data + 'lh_expression_ds01.pickle', 'rb') as f:
    lh_expression_ds01 = pickle.load(f)

##################################
# load energy gene sets dictionary
with open(path_result + 'energy_genelist_dict.pickle', 'rb') as f:
    energy_dict = pickle.load(f)

# retrieving energy expression matrices
pca = PCA(n_components=1)

energy_exp_lh = {}
energy_mean_lh = {}

# getting expression matrices, pc1 and mean maps for energy pathways
for key, value in energy_dict.items():
    energy_exp_lh[key] = geneset_expression(lh_expression_ds01, value, key, path_result)
    energy_mean_lh[key] = np.mean(energy_exp_lh[key], axis=1)

# convert dict to datframe
energy_mean_lh_df = pd.DataFrame.from_dict(energy_mean_lh, 
                                        orient='columns').reset_index(drop=True)

# keeping only left hemisphere data
# this sets the right hemisphere data as NaN and lets me plot only the left hemisphere
# for col in energy_mean_df_lh.columns:
#     energy_mean_df_lh[col] = energy_mean_df_lh[col].iloc[:200]

# saving
with open(path_result + 'energy_expression_matrix_lh.pickle', 'wb') as f:
    pickle.dump(energy_exp_lh, f)
with open(path_result + 'energy_mean_expression_lh.pickle', 'wb') as f:
    pickle.dump(energy_mean_lh, f)
with open(path_result + 'energy_pc1_expression_lh.pickle', 'wb') as f:
    pickle.dump(energy_pc1_lh, f)

# loading
with open(path_result + 'energy_expression_matrix_lh.pickle', 'rb') as f:
    energy_exp_lh = pickle.load(f)
with open(path_result + 'energy_mean_expression_lh.pickle', 'rb') as f:
    energy_mean_lh = pickle.load(f)
with open(path_result + 'energy_pc1_expression_lh.pickle', 'rb') as f:
    energy_pc1_lh = pickle.load(f)

#########################################
# comparison table for number of genes in gene list vs. ahba
size_table = pd.DataFrame(index=energy_dict.keys(),
                          columns=['gene_set', 'ahba'])
for pathway in energy_dict.keys():
    # size_table.loc[pathway, 'go'] = len(go_dict[pathway])
    # size_table.loc[pathway, 'reactome'] = len(reactome_dict[pathway])
    size_table.loc[pathway, 'gene_set'] = len(energy_dict[pathway])
    size_table.loc[pathway, 'ahba'] = energy_exp_lh[pathway].shape[1]

size_table.to_csv(path_result+'comparison_table_lh.csv')


# plotting mean gene expression for left hemisphere
for key, value in energy_mean_lh.items():
    plot_schaefer_fsaverage(value, cmap=cmaps.matter_r, hemi='L')
    plt.title(key)
    # plt.savefig(path_fig+key+'_mean_lh.svg')
    plt.show()

###################################
# cross correlation of energy maps
# main energy pathways
main_energy = ['glycolysis', 'ppp', 'tca', 'oxphos', 'lactate']
energy_mean_lh_df = energy_mean_lh_df[main_energy]

# keeping only left hemisphere data
energy_mean_lh_df = energy_mean_lh_df.iloc[:200]

# plotting
plt.figure(figsize=(3.5,3.5))
sns.heatmap(energy_mean_lh_df.corr('spearman'), annot=True, square=True,
            mask=np.tril(np.ones(energy_mean_lh_df.shape[1])),
            cmap=cmaps.BlueWhiteOrangeRed, cbar=True, linewidths=.3,
            xticklabels=energy_mean_lh_df.columns,
            yticklabels=energy_mean_lh_df.columns,
            vmin=-1, vmax=1, alpha=1)
plt.yticks(rotation=0)
plt.xticks(rotation=45)
plt.tight_layout()
# plt.savefig(path_fig+'pathway_mean_correlation_lh.svg')
plt.show()

##########################################
# correlation of energy maps with genepc1
# pc1 computed on only the left hemisphere expression matrix
pca = PCA(n_components=1)
genepc1 = np.squeeze(pca.fit_transform(pd.concat(lh_expression_schaefer400).groupby('label').mean()[:200]))

r = []
pspin = []
for key, value in energy_mean_lh.items():
        if key in main_energy:
            corr, _, p = corr_spin_test(genepc1, value.iloc[200:], spins10k[:200])
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
plt.savefig(path_fig+'energy_genepc1_corr_lh.svg')
plt.show()


# correlate to the left hemisphere of the whole brain maps
# load energy pathway data for whole brain
with open(path_result+'energy_mean_expression.pickle', 'rb') as f:
    energy_mean = pickle.load(f)

# load energy pathway mean for left hemisphere
with open(path_result+'energy_mean_expression_lh.pickle', 'rb') as f:
    energy_mean_lh = pickle.load(f)

# focusing on main energy pathways
energy_main =['glycolysis', 'ppp', 'tca',
              'oxphos', 'lactate']
# convert dict to datframe
energy_mean_df = pd.DataFrame(energy_mean).reset_index(drop=True)
energy_mean_df = energy_mean_df[energy_main]

r = []
pspin = []
for pathway in energy_mean_df.columns:
    corr, _, p = corr_spin_test(energy_mean_df[pathway].iloc[:200], 
                                energy_mean_lh_df[pathway], plot=True, spins=spins10k[:200])
    
    plt.savefig(path_fig+'sensitivity_corr_'+pathway+'.svg')
    plt.show()
    r.append(corr)
    pspin.append(p)

colors = ['sandybrown' if pspin<0.05 else 'darkgrey' for pspin in pspin]
plt.figure(figsize=(2,2.5))
plt.barh(y=energy_mean_df.columns, width=r, color=colors, height=0.4)
plt.ylabel('rho')
sns.despine()
plt.tight_layout()
plt.show()
