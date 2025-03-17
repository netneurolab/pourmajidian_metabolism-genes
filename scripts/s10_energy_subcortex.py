
'''
energy gene expression in subcortex
'''

import numpy as np
import pandas as pd
import abagen
import pickle
import nibabel as nib
import seaborn as sns
import matplotlib as mpl
import colormaps as cmaps
import matplotlib.pyplot as plt
from neuromaps.images import relabel_gifti
from neuromaps.transforms import fsaverage_to_fslr
from netneurotools import datasets, plotting, utils
from enigmatoolbox.datasets import load_summary_stats
from enigmatoolbox.utils.parcellation import parcel_to_surface
from enigmatoolbox.plotting import plot_cortical, plot_subcortical
from scipy.stats import spearmanr, zscore
from scripts.utils import corr_spin_test, geneset_expression

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update({'font.size': 8})

# path to directories
path_data = './data/'
path_result = './results/'
path_fig = './figures/'

# load energy gene sets
with open(path_result+'energy_genelist_dict.pickle', 'rb') as f:
    energy_genes = pickle.load(f)

############################################################
# fixing label order mismatch between enigma and abagen's dk

# enigma subcortical labels
sctx_enigma = load_summary_stats('22q')['SubVol_case_vs_controls']['Structure']
# 16 subcortical regions, alphabetical order, 2 lateral ventricles
# ['Laccumb', 'Lamyg', 'Lcaud', 'Lhippo', 'Lpal', 'Lput', 'Lthal',
# 'LLatVent', 'Raccumb', 'Ramyg', 'Rcaud', 'Rhippo', 'Rpal', 'Rput',
# 'Rthal', 'RLatVent'],

# abagen dk subcortical surface
#set surface=False to get volumetric parcellation with subcortex data
dk = abagen.fetch_desikan_killiany(surface=False)
#check number of rois
np.unique(nib.load(dk['image']).get_fdata()) # 84, zero is background

# labels
dk_labels = pd.read_csv(dk['info'])
sctx_labels = dk_labels[dk_labels['structure']!='cortex'] # 15 rois
#dropping brainstem
sctx_labels = sctx_labels[sctx_labels['label'] != 'brainstem'] #14 rois
sctx_labels_L = sctx_labels[sctx_labels['hemisphere']=='L'] # 7 rois
sctx_labels_R = sctx_labels[sctx_labels['hemisphere']=='R'] # 7 rois

# sort to match enigma, which is alphabetical
sctx_labels_L_sorted = sctx_labels_L.sort_values(by='label')
sctx_labels_R_sorted = sctx_labels_R.sort_values(by='label')

sctx_labels_sorted = pd.concat([sctx_labels_L_sorted, sctx_labels_R_sorted])

# final subcortical order
sctx_index = sctx_labels_sorted['id']

# the difference between abagen dk and enigma dk is that:
# enimga has lateral ventricles -> in plotting by enigma, set ventricles as False
# abagen dk has the one brainstem roi > i removed brainstem roi

# get expression data including the subcortex
expression_dk = abagen.get_expression_data(dk['image'], dk['info'],
                                                 norm_structures=True,
                                                 lr_mirror='bidirectional', 
                                                 missing='interpolate', 
                                                 return_donors=True)

rexpression01_dk, ds01 = abagen.correct.keep_stable_genes(list(expression_dk.values()), 
                                                   threshold=0.1, 
                                                   percentile=False, 
                                                   return_stability=True)

expression_dk_ds = pd.concat(rexpression01_dk).groupby('label').mean()

# save expression matrices
# save expression dict for filtering
with open(path_data + 'expression_dict_dk.pickle', 'wb') as f:
    pickle.dump(expression_dk, f)
with open(path_data + 'expression_dk_ds01.pickle', 'wb') as f:
    pickle.dump(expression_dk_ds, f)

diff_stable = {'gene': np.array(expression_dk['15496'].columns), 'ds': ds01}
diff_stable = pd.DataFrame(diff_stable)
diff_stable.to_csv(path_result+'gene_ds_all_dk.csv')

#################
# energy analysis
# load dk expression
with open(path_data + 'expression_dk_ds01.pickle', 'rb') as f:
    expression_dk_ds = pickle.load(f)

# keep subcortical expression
expression_dk_ds_sctx = expression_dk_ds.loc[sctx_index] # 14 by 13305

# retrieve energy expression data for the subcortex
energy_exp_sctx = {}
energy_mean_sctx = {}

for key, value in energy_genes.items():
    energy_exp_sctx[key] = geneset_expression(expression_dk_ds_sctx, 
                                              value, key, path_result)
    energy_mean_sctx[key] = np.mean(energy_exp_sctx[key], axis=1)

energy_main = ['glycolysis', 'ppp', 'tca', 'oxphos', 'lactate']

# setting ventricles to false cause they dont exist in abagen dk
for key, value in energy_mean_sctx.items():
    if key in energy_main:
        plot_subcortical(array_name=zscore(value), size=(800, 400),
                        cmap='Reds', color_bar=True, ventricles=False, zoom=1.2, 
                        screenshot=False, 
                        # filename=path_fig+key+'_sctx.svg',
                        transparent_bg=False,)

##########
# barplot
energy_main_sctx = pd.DataFrame(energy_mean_sctx)[energy_main].reset_index(drop=True)
energy_main_sctx['labels'] = sctx_labels_sorted['label'].reset_index(drop=True)

energy_main_sctx_lh = energy_main_sctx.iloc[:7, :]

#zscore
energy_main_sctx_lh.iloc[:, :5] = zscore(energy_main_sctx_lh.iloc[:, :5], axis=0)

# grouped barplot
energy_long = pd.melt(energy_main_sctx_lh, 
                      id_vars=['labels'],
                      value_vars=energy_main,
                      var_name='pathway',
                      value_name='expression')

# plot
plt.figure(figsize=(4, 4))
bp = sns.barplot(data=energy_long, x='expression', y='labels', 
            hue='pathway', palette='tab10',
            dodge=True, width=0.7,
            linewidth=1.4, edgecolor='w')

for bar in bp.patches:
    facecolor = bar.get_facecolor()
    bar.set_facecolor((*facecolor[:3], 0.8))
    
plt.xlabel('zscore(expression)')
# plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
plt.ylabel(None)
plt.xlim((-2,2))
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
sns.despine()
plt.tight_layout()
plt.savefig(path_fig+'energy_sctx_regions_bar.svg')
plt.show()
