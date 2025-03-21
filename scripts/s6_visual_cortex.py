
'''
Visual cortex analysis
Date: 27/11/2024
'''

import numpy as np
import pandas as pd
import pickle
import abagen
import matplotlib.pyplot as plt
import seaborn as sns
import colormaps as cmaps
import nibabel as nib
from nibabel import freesurfer
from neuromaps import images, transforms
from sklearn.decomposition import PCA
from nilearn.datasets import fetch_atlas_schaefer_2018
from scripts.utils import (plot_schaefer_fsaverage, visualize_atlas, 
                           filter_expression_ds, geneset_expression, 
                           glasser_plot, glasser_plot_roi)
from scipy.stats import zscore

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update({'font.size': 8})
plt.rcParams['lines.linewidth'] = 1

# path to directories
path_data = './data/'
path_result = './results/'
path_fig = './figures/'

# loading glasser 360 atlas annot files for plotting
glasser_lh = freesurfer.read_annot(path_data+'atlases/lh.HCPMMP1.annot')
glasser_rh = freesurfer.read_annot(path_data+'atlases/rh.HCPMMP1.annot')

# look into atlas shape
labels_lh, ctab_lh, names_lh = glasser_lh
labels_rh, ctab_rh, names_rh = glasser_rh
# 164k resolution for each hemisphere
# number of unique labels 181, 0 is background
# number of names 181

################################################
# getting visual hierarchy data and sorting them
# files are from neuroimaging core/atlases database
roi_list = pd.read_csv(path_data+'atlases/HCP-MMP1_UniqueRegionList.csv', )
roi_list = roi_list[['regionName', 'regionLongName', 'region', 'Lobe', 'cortex', 'LR']]
# make indexes start from 1 to match the glasser labels
roi_list.index += 1
roi_list = roi_list.reset_index(names='label')

# get only visual regions
vis_roi = roi_list[roi_list['cortex'].str.contains('visual', case=False)]
# 52 regions between both hemispheres
# keep left hemisphere visual rois
vis_roi_lh = vis_roi[:26]

####################
# load glasser atlas
# from brainspaces github in mni space
glasser_bspace = nib.load(path_data+'atlases/glasser360MNI.nii')
# 360 unique labels, continuous across hemispheres

##########################################
# this has 360 labels so it should be fine
glass_exp = abagen.get_expression_data(atlas=path_data+'glasser360MNI.nii',
                                        lr_mirror='bidirectional', 
                                        missing='interpolate', 
                                        return_donors=True, )

# keep stable genes
glass_exp_ds, ds = filter_expression_ds(glass_exp, ds=0.1)
# 360 x 9551

# save expression dict for filtering
with open(path_data + 'expression_dict_glasser.pickle', 'wb') as f:
    pickle.dump(glass_exp, f)

with open(path_data + 'expression_glasser_ds01.pickle', 'wb') as f:
    pickle.dump(glass_exp_ds, f)

############################
# load expression dictionary
with open(path_data + 'expression_dict_glasser.pickle', 'rb') as f:
    glass_exp = pickle.load(f)

with open(path_data + 'expression_glasser_ds01.pickle', 'rb') as f:
    glass_exp_ds = pickle.load(f)

# load energy gene sets
with open(path_result + 'energy_genelist_dict.pickle', 'rb') as f:
    energy_dict = pickle.load(f)

# retrieving energy expression matrices for glasser
energy_exp = {}
energy_mean = {}

# getting expression matrices, pc1 and mean maps for energy pathways
for key, value in energy_dict.items():
    energy_exp[key] = geneset_expression(glass_exp_ds, value, key, path_result)
    energy_mean[key] = np.mean(energy_exp[key], axis=1)

# saving
with open(path_result + 'energy_exp_matrix_glasser.pickle', 'wb') as f:
    pickle.dump(energy_exp, f)
with open(path_result + 'energy_mean_exp_glasser.pickle', 'wb') as f:
    pickle.dump(energy_mean, f)

##############
# run analysis

# load energy expression and mean maps
with open(path_result + 'energy_exp_matrix_glasser.pickle', 'rb') as f:
    energy_exp = pickle.load(f)
with open(path_result + 'energy_mean_exp_glasser.pickle', 'rb') as f:
    energy_mean = pickle.load(f)

energy_main = ['glycolysis', 'ppp', 'tca', 'oxphos', 'lactate']

energy_mean = pd.DataFrame(energy_mean)
energy_mean = energy_mean[energy_main]

# plotting mean gene expression whole brain glasser
for pathway in energy_mean.columns:
        glasser_plot(energy_mean[pathway], 
                     outlinecolor='BuPu', 
                     brightness=0.9,
                     views=['lateral', 'medial']
                    )
        plt.title(pathway)
        # plt.savefig(path_fig+pathway+'_glasser.svg', dpi=800)
        plt.show()


# oxphos map

# plotting mean gene expression vis_lh only
names_lh_df = pd.DataFrame(names_lh, columns=['roi'])

# get names of regions to drop
to_drop = names_lh_df[~names_lh_df.index.isin(vis_roi_lh['label'])]
to_drop_rois = to_drop['roi'].to_list()
to_drop_all = to_drop_rois + names_rh

# plot
glasser_plot(energy_mean['oxphos'][vis_roi_lh['label']], 
                outlinecolor='BuPu', 
                roi=True, 
                roi_drop=to_drop_all,
                brightness=1,
                hemi='L'
            )
plt.title('oxphos')
plt.savefig(path_fig+'oxphos_glasser_vis.svg', dpi=800)
plt.show()

# interactive posterior view for better angle
glasser_plot(energy_mean['oxphos'][vis_roi_lh['label']], 
                outlinecolor='BuPu', 
                roi=True, 
                roi_drop=to_drop_all,
                brightness=1,
                hemi='L',
                views='posterior',
                interactive=True,)
plt.show()


# sort based on info process hierarchy for plots
np.unique(vis_roi['cortex'])
# ['Dorsal_Stream_Visual', 'Early_Visual', 'MT+_Complex_and_Neighboring_Visual_Areas', 
# 'Primary_Visual', 'Ventral_Stream_Visual']
order1 = ['V1', 
          'V2', 'V3', 'V4',
          'V3A', 'V3B', 'V6', 'V6A', 'V7', 'IPS1',
          'V8', 'VVC', 'PIT', 'FFC', 'VMV1', 'VMV2', 'VMV3',
          'V3CD', 'LO1', 'LO2', 'LO3', 'V4t', 'FST', 'MT', 'MST', 'PH']

order2 = ['Primary_Visual', 'Early_Visual', 'Dorsal_Stream_Visual',
          'Ventral_Stream_Visual', 'MT+_Complex_and_Neighboring_Visual_Areas']

# plot each region to know what each parcel is
for roi in order1:
    glasser_plot_roi(energy_mean['oxphos'], 
                outlinecolor='BuPu', 
                roi=True, 
                roi_idx=vis_roi_lh[vis_roi_lh['region']==roi]['label'],
                brightness=0.9,
                hemi='L',)
    plt.title(roi)
    plt.show()


energy_df = pd.DataFrame(energy_mean)[energy_main]

energy_df = pd.concat([roi_list, energy_df.reset_index()], axis=1)
energy_df_vis = energy_df[energy_df['cortex'].str.contains('visual', case=False)]
# keep left hemisphere
energy_df_vis_lh = energy_df_vis.iloc[:26]

# sort based on broad hierarchy
energy_df_vis = energy_df_vis_lh.copy()
energy_df_vis['cortex'] = pd.Categorical(energy_df_vis['cortex'], categories=order2, ordered=True)
energy_df_vis_sorted = energy_df_vis.sort_values('cortex')

# sort rois
# energy_df_vis['region'] = pd.Categorical(energy_df_vis['region'], categories=order1, ordered=True)
# energy_df_vis_sorted = energy_df_vis.sort_values('region')

energy_df_vis_sorted = energy_df_vis_sorted[['glycolysis', 'ppp', 'tca', 
                                             'oxphos', 'lactate', 
                                             'cortex', 'region']]

# average in each hierarchy level
energy_df_vis_mean = energy_df_vis_sorted.groupby('cortex').mean()

# plot
n = len(energy_main)
fig, axes = plt.subplots(1, n, figsize=(1.1 * n, 2), sharey=True)

for i, pathway in enumerate(energy_main):
    axes[i].bar(['primary', 'early', 'dorsal', 'ventral', 'mt+'], 
           zscore(energy_df_vis_mean[pathway]), 
           alpha=0.7, 
           color='sandybrown', 
           width=0.7)
    axes[i].set_xticks(range(5))
    axes[i].set_xticklabels(['primary', 'early', 'dorsal', 'ventral', 'mt+'], rotation=90)
    axes[i].set_title(pathway)
    axes[i].set_ylim((-3, 3))
    if i == 0:
        axes[i].set_ylabel('zscore(expression)')
    sns.despine()

plt.tight_layout()
# plt.savefig(path_fig+'visual_hier2.svg')
plt.show()


# plot all the rois in each hierarchy level
for pathway in energy_main:
    plt.figure(figsize=(4, 3))
    plt.bar(zscore(energy_df_vis_sorted['region']), 
            energy_df_vis_sorted[pathway], 
            alpha=0.7, 
            color = 'sandybrown',
            width=0.7)
    plt.xticks(rotation=90)
    plt.ylabel('zscore(expression)')
    plt.title(pathway)
    sns.despine()
    plt.margins(x=0.02)
    # plt.ylim((-3, 3))
    plt.tight_layout()
    # plt.savefig(path_fig+pathway+'_visual_hier1.svg')
    plt.show()
