
'''
Plotting and some initial exploration 
of energy pathway maps

1. Plot energy maps on brain
2. Map-map correlations
3. Correlated gene expression 


Author: Moohebat
Date: 19/06/2024
'''

import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import colormaps as cmaps
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import zscore
from scripts.utils import corr_spin_test, pair_corr_spin, plot_schaefer_fsaverage
import matplotlib.patches as patches
from netneurotools import plotting

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
spins10k = np.load(path_data+'spins10k.npy')

#
# plot mean gene expression
for key, value in energy_mean.items():
    plot_schaefer_fsaverage(zscore(value), cmap=cmaps.matter_r)
    plt.title(key)
    plt.savefig(path_fig+key+'_mean.svg')
    # plt.show()

# plot pc1 gene expression
for key, value in energy_pc1.items():
    plot_schaefer_fsaverage(-value, cmap=cmaps.BlueWhiteOrangeRed)
    plt.title(key)
    plt.savefig(path_fig+key+'_pc1.svg')
    # plt.show()

########################################################
# from this point on I focus on the main energy pathways 

# correlation of energy maps
energy_mean_df = pd.DataFrame.from_dict(energy_mean, 
                                        orient='columns').reset_index(drop=True)

main_energy = ['glycolysis', 'ppp', 'tca', 'oxphos', 'lactate']
energy_mean_df = energy_mean_df[main_energy]

# calculate pairwise correlation between maps with spin test
energy_corr, energy_pspin = pair_corr_spin(energy_mean_df, energy_mean_df, spins10k)

# plotting
plt.figure(figsize=(4,4))
mask = np.tril(np.ones(energy_mean_df.shape[1]))
ax = sns.heatmap(energy_corr, annot=True, square=True,
            mask=mask,
            cmap=cmaps.BlueWhiteOrangeRed, cbar=True, linewidths=.4,
            linecolor='darkgrey',
            xticklabels=energy_mean_df.columns,
            yticklabels=energy_mean_df.columns,
            vmin=-1, vmax=1, alpha=1)

# # to add significance 
# for i in range(energy_pspin.shape[0]):
#     for j in range(energy_pspin.shape[1]):
#         if mask[i, j] and i != j:
#             if energy_pspin.iloc[i, j] < 0.05:
#                 ax.text(i+0.6, j+0.4, '*', ha='left', va='bottom', color='k')
plt.yticks(rotation=0)
plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
plt.tight_layout()
plt.savefig(path_fig+'pathway_mean_correlation.svg')
plt.show()

#########################################
# correlated gene expression and plotting

# keeping only main energy pathway expressions
energy_exp = {k: energy_exp[k] for k in main_energy}

# concatenate all energy expression matrices
exp_matrix = pd.concat(energy_exp.values(), axis=1)

exp_matrix = exp_matrix.reset_index(drop=True)

i_glyco = []
i_ppp = []
i_tca = []
i_oxphos = []
i_lactate = []

current_idx = 0
for pathway in main_energy:
    n_genes = len(energy_exp[pathway].columns)
    pathway_indices = list(range(current_idx, current_idx + n_genes))
    
    if pathway == 'glycolysis':
        i_glyco = pathway_indices
    elif pathway == 'ppp':
        i_ppp = pathway_indices
    elif pathway == 'tca':
        i_tca = pathway_indices
    elif pathway == 'oxphos':
        i_oxphos = pathway_indices
    elif pathway == 'lactate':
        i_lactate = pathway_indices
    
    current_idx += n_genes

# convert to numpy arrays
i_glyco = np.array(i_glyco)
i_ppp = np.array(i_ppp)
i_tca = np.array(i_tca)
i_oxphos = np.array(i_oxphos)
i_lactate = np.array(i_lactate)

# define classes for communities
classes = np.zeros((len(exp_matrix.columns), 1))
classes[i_glyco, 0] = 1
classes[i_ppp, 0] = 2
classes[i_tca, 0] = 3
classes[i_oxphos, 0] = 5
classes[i_lactate, 0] = 6

# plotting
class_names = ['glycolysis', 'ppp', 'tca', 'oxphos', 'lactate']
fig, axs = plt.subplots(1, 1, figsize=(17,17))

inds = plotting.sort_communities(np.corrcoef(zscore(exp_matrix).T), classes[:, 0])
bounds = plotting._grid_communities(classes[:, 0])

sns.heatmap(data=np.corrcoef(zscore(exp_matrix).T)[np.ix_(inds, inds)],
            vmin=-1, vmax=1, ax=axs, cbar=True, square=True, cmap=cmaps.BlueWhiteOrangeRed,
            linewidths=0, xticklabels=exp_matrix.columns[inds], yticklabels=exp_matrix.columns[inds])

for n, edge in enumerate(np.diff(bounds)):
    axs.add_patch(patches.Rectangle((bounds[n], bounds[n]), 
                                        edge, edge, fill=False,
                                        linewidth=2, edgecolor='black'))

plt.tight_layout()
plt.savefig(path_fig+'energy_correlated_geneexp_communit_sorted.svg')
plt.show()


#############################################################
# correlation of energy maps with pc1 and average expression

with open(path_data+'expression_dict_schaefer400.pickle', 'rb') as f:
    expression_dict_schaefer400 = pickle.load(f)

expression_schaefer400 = pd.concat(expression_dict_schaefer400).groupby('label').mean()

pca = PCA(n_components=1)
genepc1 = np.squeeze(pca.fit_transform(expression_schaefer400))
gene_avg = np.mean(expression_schaefer400, axis=1).reset_index(drop=True)

# correlation with ahba pc1 of gene expression
r = []
pspin = []
for column in energy_mean_df.columns:
        corr, _, p = corr_spin_test(genepc1, energy_mean_df[column], spins10k)
        r.append(corr)
        pspin.append(p)

# plot
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
plt.savefig(path_fig+'energy_genepc1_corr.svg')
plt.show()


# correlation with ahba mean gene expression
r = []
pspin = []
for column in energy_mean_df.columns:
        corr, _, p = corr_spin_test(gene_avg, energy_mean_df[column], spins10k)
        r.append(corr)
        pspin.append(p)

# plot
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
plt.title('corr with average expression')
# plt.autoscale(enable=True, axis='both', tight=True)
plt.tight_layout()
sns.despine()
plt.savefig(path_fig+'energy_geneavg_corr.svg')
plt.show()


##############################################################
# exploring principal components of energy expression matrices
pca = PCA(n_components=5)

pc_dict = {}
pc_varexp = {}
fig, axs = plt.subplots(1, 5, figsize=(5, 1.5), sharey=True, sharex=True)
plot_number = 1
for key, value in energy_exp.items():
    if key in main_energy:
        pc1_exp = np.squeeze(pca.fit_transform(value))
        pc_dict[key] = pc1_exp
        pc_varexp[key] = pca.explained_variance_ratio_ * 100

        ax = plt.subplot(1, 5, plot_number)
        sns.scatterplot(pca.explained_variance_ratio_*100, color='sandybrown', ax=ax)
        plt.xticks(ticks=np.arange(5), labels=['1', '2', '3', '4', '5'])
        plt.xlabel('PC')
        plt.yticks(ticks=np.arange(0,70,10), labels=['0', '10', '20', '30', '40', '50', '60'])
        plt.ylabel('Variance explained (%)')
        plt.title(key)
        sns.despine()
        plt.tight_layout()

        plot_number += 1
# fig.text(0, 0.5, 'Variance explained (%)', va='center', rotation='vertical', )
plt.tight_layout()
plt.savefig(path_fig+'energy_pca_varexp.svg')
plt.show()

#save the first 5 principal components for future analysis
with open(path_result + 'energy_5pc_dict.pickle', 'wb') as f:
    pickle.dump(pc_dict, f)

# variance explained
# 'glycolysis': array([45.46565715, 28.37404844,  7.71548844,  5.12357556,  4.02646495])
# 'ppp': array([32.30690707, 26.27980368, 21.17209168, 12.16870863,  8.07248895])
# 'tca': array([44.60065014, 27.00175601,  6.81429161,  4.32698463,  4.22715682])
# 'oxphos': array([55.01795986, 20.89395465,  6.18222829,  3.35700115,  2.45578734])
# 'lactate': array([49.55962407, 23.66118258,  6.31637238,  4.67088316,  3.99285252])

########################
# plotting the PGD gene
plot_schaefer_fsaverage(zscore(energy_exp['ppp']['PGD']), cmap=cmaps.matter_r)
plt.title('PGD')
plt.savefig(path_fig+'PGD.svg')
plt.show()

# correlations with energy maps
for pathway in main_energy:
    corr, _, p_spin = corr_spin_test(energy_exp[pathway]['PGD'].reset_index(drop=True), 
                                            energy_mean_df[pathway], 
                                            spins10k,
                                            scattercolor='sandybrown',
                                            linecolor = 'grey',
                                            plot=True)
    plt.savefig(path_fig+pathway+'_PGD_corr.svg')
    plt.show()