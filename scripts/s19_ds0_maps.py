
'''
Repeat all the main analysis without applying the 
differential stability threshold to AHBA data

Author: Moohebat
'''
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import colormaps as cmaps
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.stats import zscore
from scripts.utils import (geneset_expression,
                           corr_spin_test, 
                           pair_corr_spin,
                           class_enrichment,
                           plot_schaefer_fsaverage)

# path to directories
path_data = './data/'
path_result = './results/'
path_fig = './figures/'

# load data
with open(path_data + 'expression_dict_schaefer400.pickle', 'rb') as f:
    expression_dict_schaefer400 = pickle.load(f)

with open(path_result + 'energy_genelist_dict.pickle', 'rb') as f:
    energy_dict = pickle.load(f)

# loading schaefer400 spins
spins10k = np.load(path_data+'spins10k.npy')


##########
# analysis

expression_schaefer400 = pd.concat(expression_dict_schaefer400).groupby('label').mean()
# 400 x 15633

# retrieving energy expression matrices
energy_exp = {}
energy_mean = {}
energy_median = {}
energy_pc1 = {}

pca = PCA(n_components=1)

# getting expression matrices, pc1 and mean maps for energy pathways
for key, value in energy_dict.items():
    energy_exp[key] = geneset_expression(expression_schaefer400, value, key, path_result)
    energy_mean[key] = np.mean(energy_exp[key], axis=1)
    energy_median[key] = np.median(energy_exp[key], axis=1)
    energy_pc1[key] = np.squeeze(pca.fit_transform(energy_exp[key]))

'''
glycolysis
['ADPGK', 'ALDOA', 'ALDOC', 'BPGM', 'ENO1', 'ENO2', 'ENO3', 'ENO4',
       'GAPDH', 'GAPDHS', 'GCK', 'GPI', 'PFKFB2', 'PFKL', 'PFKM', 'PFKP',
       'PGAM1', 'PGAM2', 'PGK1', 'PPP2CA', 'PRKACA', 'TPI1']

ppp
['DERA', 'G6PD', 'PGD', 'PGLS', 'PRPS2', 'RBKS', 'RPE', 'RPEL1', 'RPIA',
       'SHPK', 'TALDO1', 'TKT']

tca
['ACO2', 'CS', 'DLST', 'FH', 'IDH2', 'IDH3A', 'IDH3B', 'IDH3G', 'MDH2',
       'NNT', 'OGDH', 'SDHA', 'SDHB', 'SDHC', 'SDHD', 'SUCLA2', 'SUCLG1',
       'SUCLG2']

oxphos
['ATP5F1A', 'ATP5F1B', 'ATP5F1C', 'ATP5F1D', 'ATP5ME', 'ATP5MF',
       'ATP5MG', 'ATP5PB', 'ATP5PD', 'ATP5PF', 'ATP5PO', 'COX4I1', 'COX5A',
       'COX5B', 'COX6A1', 'COX6B1', 'COX6C', 'COX7A2L', 'COX7B', 'COX7C',
       'COX8A', 'CYC1', 'CYCS', 'NDUFA1', 'NDUFA10', 'NDUFA2', 'NDUFA3',
       'NDUFA4', 'NDUFA5', 'NDUFA6', 'NDUFA7', 'NDUFA8', 'NDUFA9', 'NDUFAB1',
       'NDUFAF1', 'NDUFB1', 'NDUFB10', 'NDUFB2', 'NDUFB3', 'NDUFB4', 'NDUFB5',
       'NDUFB6', 'NDUFB7', 'NDUFB8', 'NDUFB9', 'NDUFC1', 'NDUFC2', 'NDUFS1',
       'NDUFS2', 'NDUFS3', 'NDUFS4', 'NDUFS5', 'NDUFS6', 'NDUFS7', 'NDUFS8',
       'NDUFV1', 'NDUFV2', 'NDUFV3', 'SDHA', 'SDHB', 'SDHC', 'SDHD', 'UQCR10',
       'UQCR11', 'UQCRB', 'UQCRC1', 'UQCRC2', 'UQCRFS1', 'UQCRH', 'UQCRQ']

lactate
['ACACB', 'EMB', 'GATD1', 'HAGH', 'HAGHL', 'HIF1A', 'LDHA', 'LDHAL6A',
       'LDHB', 'LDHC', 'LDHD', 'MRS2', 'PARK7', 'PER2', 'PFKFB2', 'PNKD',
       'SLC16A1', 'SLC16A3', 'SLC16A7', 'SLC16A8', 'SLC37A4', 'SLC5A12',
       'TIGAR']
'''

# save
with open(path_result + 'ds0_energy_exp_matrix.pickle', 'wb') as f:
    pickle.dump(energy_exp, f)
with open(path_result + 'ds0_energy_mean_exp.pickle', 'wb') as f:
    pickle.dump(energy_mean, f)
with open(path_result + 'ds0_energy_pc1_exp.pickle', 'wb') as f:
    pickle.dump(energy_pc1, f)
with open(path_result + 'ds0_energy_median_exp.pickle', 'wb') as f:
    pickle.dump(energy_median, f)

# plot
for key, value in energy_mean.items():
    plot_schaefer_fsaverage(zscore(value), cmap=cmaps.matter_r)
    plt.title(key)
    plt.savefig(path_fig+key+'_ds0.svg', dpi=600)
    plt.show()

# convert dict to datframe
energy_mean_df = pd.DataFrame(energy_mean).reset_index(drop=True)
energy_main = ['glycolysis', 'ppp', 'tca', 'oxphos', 'lactate']
energy_mean_df = energy_mean_df[energy_main]

# plot map-map correlations
energy_corr, energy_pspin = pair_corr_spin(energy_mean_df, energy_mean_df, spins10k)

plt.figure(figsize=(4,4))
mask = np.tril(np.ones(energy_mean_df.shape[1]))
ax = sns.heatmap(energy_corr, annot=True, square=True,
            mask=mask,
            cmap=cmaps.BlueWhiteOrangeRed, cbar=True, linewidths=.4,
            xticklabels=energy_mean_df.columns,
            yticklabels=energy_mean_df.columns,
            vmin=-1, vmax=1, alpha=1)
# # to add significance level
# for i in range(energy_pspin.shape[0]):
#     for j in range(energy_pspin.shape[1]):
#         if mask[i, j] and i != j:
#             if energy_pspin.iloc[i, j] <= 0.05:
#                 ax.text(i+0.6, j+0.4, '*', ha='left', va='bottom', color='k')
plt.yticks(rotation=0)
plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
plt.title('map-map correlations - ds0')
plt.tight_layout()
plt.savefig(path_fig+'energy_mean_corr_ds0.svg')
plt.show()

# compare with original energy mean maps
with open(path_result+'energy_mean_expression.pickle', 'rb') as f:
    energy_mean_orig = pickle.load(f)

energy_mean_orig_df = pd.DataFrame(energy_mean_orig).reset_index(drop=True)

# just keep main pathways
energy_mean_orig_df = energy_mean_orig_df[energy_main]

for pathway in energy_main:
    corr_spin_test(energy_mean_orig_df[pathway], energy_mean_df[pathway],
                   spins10k, plot=True)
    plt.xlabel(pathway+' original')
    plt.ylabel(pathway+' ds0')
    plt.tight_layout()
    plt.savefig(path_fig+pathway+'_ds0_orig_corr.svg')
    plt.show()
               
    
# correlation of energy maps with pc1 expression and average expression
with open(path_data+'expression_dict_schaefer400.pickle', 'rb') as f:
    expression_dict_schaefer400 = pickle.load(f)

expression_schaefer400 = pd.concat(expression_dict_schaefer400).groupby('label').mean()

pca = PCA(n_components=1)
genepc1 = np.squeeze(pca.fit_transform(expression_schaefer400))
gene_avg = np.mean(expression_schaefer400, axis=1).reset_index(drop=True)

r = []
pspin = []
for column in energy_mean_df.columns:
        corr, _, p = corr_spin_test(genepc1, energy_mean_df[column], spins10k)
        r.append(corr)
        pspin.append(p)

'''
r
[-0.17586353664710402, 0.5021813886336789, 0.33904299401871263, -0.12629478934243338, 0.29445784036150224]
pspin
[0.5784421557844216, 9.999000099990002e-05, 0.40535946405359463, 0.6787321267873213, 0.46015398460153983]
'''

plt.figure(figsize=(2.5,2))
bp = sns.barplot(x=r, y=energy_mean_df.columns, width=0.5, orient='h')
for i, patch in enumerate(bp.patches):
    if pspin[i] < 0.05:
        patch.set_facecolor('sandybrown')
    else:
        patch.set_facecolor('lightgrey')
# plt.xticks(rotation=45)
plt.xlim(-1,1)
plt.xlabel('rho')
plt.ylabel('pathway')
plt.title('corr with ahba genepc1')
# plt.autoscale(enable=True, axis='both', tight=True)
plt.tight_layout()
sns.despine()
plt.savefig(path_fig+'ds0_energy_genepc1_corr.svg')
plt.show()

# corr with mean ahba expression
r = []
pspin = []
for column in energy_mean_df.columns:
        corr, _, p = corr_spin_test(gene_avg, energy_mean_df[column], spins10k)
        r.append(corr)
        pspin.append(p)

'''
r
[0.45558884743029643, 0.39089663060394125, 0.23214407590047434, 0.3822713891961825, 0.4441484634278963]
pspin
[9.999000099990002e-05, 9.999000099990002e-05, 0.12968703129687031, 9.999000099990002e-05, 9.999000099990002e-05]
'''

plt.figure(figsize=(2.5,2))
bp = sns.barplot(x=r, y=energy_mean_df.columns, width=0.5, orient='h')
for i, patch in enumerate(bp.patches):
    if pspin[i] < 0.05:
        patch.set_facecolor('sandybrown')
    else:
        patch.set_facecolor('lightgrey')
# plt.xticks(rotation=45)
plt.xlim(-1,1)
plt.xlabel('rho')
plt.ylabel('pathway')
plt.title('corr with ahba average expression')
# plt.autoscale(enable=True, axis='both', tight=True)
plt.tight_layout()
sns.despine()
plt.savefig(path_fig+'ds0_energy_geneavg_corr.svg')
plt.show()


##################
# class enrichment

# loading class labels
yeo_schaefer400 = np.load(path_data+'yeo_schaefer400.npy')
ve_schaefer400 = np.load(path_data+'ve_schaefer400.npy', allow_pickle=True)
mesulam_schaefer400 = np.load(path_data+'mesulam_schaefer400.npy', allow_pickle=True)

# convert label names
# Mapping dictionary
mapping = {
    'Cont': 'FP',
    'Default': 'DM',
    'DorsAttn': 'DA',
    'Limbic': 'Lim',
    'SalVentAttn': 'SA',
    'SomMot': 'SM',
    'Vis': 'Vis'
}
# Convert the values using the mapping dictionary
yeo_schaefer400 = np.array([mapping[val] for val in yeo_schaefer400])

## ve
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

## mesulam
mapping = {
    'hetermodal': 'HM', 
    'idiotypic': 'ID', 
    'paralimbic': 'PLB', 
    'unimodal': 'UM'
}
# Convert the values using the mapping dictionary
mesulam_schaefer400 = np.array([mapping[val] for val in mesulam_schaefer400])

classes = {'yeo':{
    'order': ['SM','Vis', 'DA', 'SA', 'FP','DM','Lim'], 
    'map': yeo_schaefer400},
    've': {
        'order': ['PM','PS','PSS', 'Ac','Ac2','Ins','Lim'],
        'map': ve_schaefer400},
    'mesulam': {
        'order': ['ID', 'UM', 'HM', 'PLB'],
        'map': mesulam_schaefer400}}

class_enrichment(energy_mean_df, 
                 yeo_schaefer400, 
                 spins10k,
                 classes['yeo']['order'],
                 path_fig,
                 'yeo_enrich_ds0')

class_enrichment(energy_mean_df, 
                 ve_schaefer400, 
                 spins10k,
                 classes['ve']['order'],
                 path_fig,
                 've_enrich_ds0')

class_enrichment(energy_mean_df, 
                 mesulam_schaefer400, 
                 spins10k,
                 classes['mesulam']['order'],
                 path_fig,
                 'mesulam_enrich_ds0')


#######################################
# correlation with functional gradients
fc1_map = np.load(path_data+'fc1_margulies.npy')
fc2_map = np.load(path_data+'fc2_margulies.npy')
fc3_map = np.load(path_data+'fc3_margulies.npy')

fcs = [fc1_map, fc2_map, fc3_map]

energy_main =['glycolysis', 'ppp', 'tca',
              'oxphos', 'lactate']
energy_mean = {key: energy_mean[key] for key in energy_main}

# correlations with energy mean maps
fc1 = pd.DataFrame(index=np.arange(5), columns=['pathway', 'r', 'pspin'])
fc2 = pd.DataFrame(index=np.arange(5), columns=['pathway', 'r', 'pspin'])
fc3 = pd.DataFrame(index=np.arange(5), columns=['pathway', 'r', 'pspin'])
fc_dict = {'fc1': fc1, 'fc2': fc2, 'fc3': fc3}

margul = np.array(fcs).T
for j, (k, v) in enumerate(fc_dict.items()):
    print(j,k)
    for i, (key, value) in enumerate(energy_mean.items()):
        v['pathway'][i] = key
        v['r'][i], _, v['pspin'][i] = corr_spin_test(np.array(energy_mean[key]), 
                                                                       -margul[:, j], spins10k)
        
 
# barh plot
fig, axs = plt.subplots(1, len(fc_dict), figsize=(1.3*len(fc_dict), 1.5))
for ax, (key, value) in zip(axs, fc_dict.items()):
    colors = ['sandybrown' if pspin<0.05 else 'lightgrey' for pspin in value['pspin'][::-1]]
    ax.barh(y=value['pathway'][::-1], width=value['r'][::-1], color=colors, height=0.5, alpha=0.8)
    ax.set_xlim(-0.7,0.7)
    ax.set_title(key)
    ax.set_xlabel('rho')
    ax.set_yticks(value['pathway'][::-1])
    sns.despine()

for i, ax in enumerate(axs):
    if i != 0:
        ax.set_yticklabels(['']*len(ax.get_yticks()))
    if i == 0:    
        ax.set_ylabel('mean expression')
plt.tight_layout()
plt.savefig(path_fig+'fc_correlations_ds0.svg')
plt.show()
