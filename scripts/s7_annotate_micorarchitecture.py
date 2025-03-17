

'''
script to perform analysis for 
characterizing energy maps with brain architecture
univariate correlations
Author: Moohebat
Date: 28/06/2024
'''

#importing packages
import pickle
import numpy as np
import pandas as pd
from scipy.stats import zscore, pearsonr, spearmanr
from scripts.utils import corr_spin_test, pair_corr_spin, plot_heatmap, plot_schaefer_fsaverage
from statsmodels.stats.multitest import multipletests
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import colormaps as cmaps

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

# focusing on main energy pathways
energy_main =['glycolysis', 'ppp', 'tca', 'oxphos', 'lactate']
energy_mean_df = pd.DataFrame(energy_mean).reset_index(drop=True)
energy_df = energy_mean_df[energy_main]

# loading spins
spins10k = np.load(path_data+'spins10k.npy')

# loading maps
# pet data
with open(path_data+'pet_df.pickle', 'rb') as f:
    pet_df = pickle.load(f)
pet_df = pet_df.drop('cbv', axis=1)

# meg data
with open(path_data+'meg_df.pickle', 'rb') as f:
    meg_df = pickle.load(f)
meg_df = meg_df.drop(['meg_timescale'], axis=1)

# reorder based on frequency
meg_df = meg_df[['meg_delta', 'meg_theta', 'meg_alpha', 
                 'meg_beta', 'meg_lowgamma', 'meg_highgamma']]

# connectivity data
with open(path_data+'conn_df.pickle', 'rb') as f:
    conn_df = pickle.load(f)
conn_df = conn_df[['sc_degree', 'fc_strength']]

# wagstyl genesets, layers and cells gene markers
with open(path_result + 'wag_genesets_df.pickle', 'rb') as f:
    cell_layer_df = pickle.load(f)

# other maps, ucbj, myelin, ct, evoexp
with open(path_data + 'other_maps.pickle', 'rb') as f:
    other_maps = pickle.load(f)

##################
# running analysis
#big correlation heatmap
all_maps = pd.concat([pet_df, meg_df, cell_layer_df, conn_df], axis=1)
all_energy_corr, all_energy_pspin = pair_corr_spin(energy_df, all_maps, spins10k)

all_energy_corr.to_csv(path_result+'figure5_correlation_values.csv')
all_energy_pspin.to_csv(path_result+'figure5_pspins.csv')


# fdr correction for multiple testing
model_pval = multipletests(all_energy_pspin.values.flatten(), method='fdr_bh')[1]
model_pval = pd.DataFrame(model_pval.reshape(22,5))
model_pval.columns = all_energy_pspin.columns
model_pval.index  = all_energy_pspin.index

model_pval.to_csv(path_result+'figure5_pspins_fdr_bh.csv')

# plot heatmap and save
plt.figure(figsize=(5,2.5))
plot_heatmap(all_energy_corr, model_pval)
plt.savefig(path_fig+'all_corr_heatmap_layer_burt.svg')
plt.show()


################################
# gene-wise correlation with pet
# make energy gene list into longform
energy_exp_main = {key: energy_exp[key] for key in energy_main}
energy_exp_main = {key: energy_exp_main[key] for key in energy_main}
energy_exp_df = pd.concat(energy_exp_main.values(), axis=1)
energy_exp_df = energy_exp_df.T.drop_duplicates(keep='last').T

gene_pet_df = pd.DataFrame(columns=['gene', 'cmrglc_rho', 'key'])
gene_pet_df['gene'] = energy_exp_df.columns
for i, gene in enumerate(gene_pet_df['gene']):
    gene_pet_df['cmrglc_rho'][i] = spearmanr(np.array(energy_exp_df[gene]), 
                                             np.array(pet_df['cmrglc']))[0]
    for key, value in energy_exp_main.items():
        if gene in value.columns:
            gene_pet_df['key'][i] = key

gene_pet_df['cmrglc_rho'] = pd.to_numeric(gene_pet_df['cmrglc_rho'])

# to highlight rate limiting enzymes
rl_list = ['PFKL', 'PFKM', 'PFKP', 
           'G6PD', 'PGD', 'PGLS',
           'CS', 'IDH2', 'IDH3A', 'IDH3G', 'OGDH', 'OGDHL',
           'COX4I1', 'COX6A1', 'COX6A2', 'COX7A1', 'ATP5F1A',  'ATP5F1B',
           'LDHA', 'LDHB', 'LDHC', 'LDHD']

# plotting
plt.figure(figsize=(4,4))
sns.stripplot(data=gene_pet_df[gene_pet_df.gene.isin(rl_list)==False], 
              x='key', y='cmrglc_rho', 
              color='skyblue', jitter=True, 
              alpha=0.7, size=7, zorder=1)

# plot rate-limiting genes
rl_dots = sns.stripplot(data=gene_pet_df[gene_pet_df.gene.isin(rl_list)==True],
                         x='key', y='cmrglc_rho', 
                         color='tomato', jitter=True, 
                         alpha=0.7, size=7, zorder=2)

red_subset = gene_pet_df[gene_pet_df.gene.isin(rl_list)==True]

# annotate gene names
for line in range(0, red_subset.shape[0]):
    rl_dots.annotate(red_subset.gene.iloc[line], 
                    (red_subset.key.iloc[line], 
                     red_subset.cmrglc_rho.iloc[line]), 
                    textcoords="offset points",
                    xytext=(10,0), 
                    ha='left', va='top', size='xx-small', 
                    color='black', weight='regular',)
    
sns.despine()
plt.xlabel('energy pathway genes')
plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
plt.ylabel('rho')
plt.title('gene-cmrglc correlation')
plt.tight_layout()
plt.savefig(path_fig+'gene_cmrglc_corr_annotated.svg')
plt.show()

################
# myelin and ppp
for pathway in ['ppp', 'tca']:
    corr, _, p_spin = corr_spin_test(energy_df[pathway], 
                                            other_maps['myelin'], 
                                            spins10k,
                                            scattercolor='sandybrown',
                                            linecolor = 'grey',
                                            plot=True)
    plt.ylabel('myelin')
    plt.tight_layout()
    # plt.savefig(path_fig+pathway+'_myelin_corr.svg')
    plt.show()

# plot myelin
plot_schaefer_fsaverage(other_maps['myelin'], cmap=cmaps.matter_r)
plt.title('myelin')
plt.savefig(path_fig+'myelin.svg', dpi=800)
plt.show()