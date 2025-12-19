
'''
Parcellate mitomaps from mosharov et al., 2025
and correlate with mitochondrial energy maps

Author: Moohebat
'''
import pickle
import numpy as np
import pandas as pd
from neuromaps.parcellate import Parcellater
from nilearn.datasets import fetch_atlas_schaefer_2018
import matplotlib.pyplot as plt
import colormaps as cmaps
from statsmodels.stats.multitest import multipletests
from scripts.utils import (corr_spin_test, plot_schaefer_fsaverage, 
                           pair_corr_spin)

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update({'font.size': 8})

# path to directories
path_data = './data/'
path_result = './results/'
path_fig = './figures/'

# loading spins
spins10k = np.load(path_data+'spins10k.npy')

# load energy maps
with open(path_result+'energy_mean_expression.pickle', 'rb') as f:
    energy_mean = pickle.load(f)

# parcellate mitochonridal maps from Mosharov et al
# mni152, 1 mm
schaefer400 = fetch_atlas_schaefer_2018() #MNI152, 1mm
parc_mni152 = Parcellater(schaefer400['maps'], 'MNI152')


c1 = path_data + "CI.nii.gz"
c1_400 = np.squeeze(parc_mni152.fit_transform(c1, 'MNI152', ignore_background_data=True).T)

c2 = path_data + "CII.nii.gz"
c2_400 = np.squeeze(parc_mni152.fit_transform(c2, 'MNI152', ignore_background_data=True).T)

c4 = path_data +'CIV.nii.gz' # mni152, 1 mm
c4_400 = np.squeeze(parc_mni152.fit_transform(c4, 'MNI152', ignore_background_data=True).T)

mitod = path_data+"MitoD.nii.gz"
mitod_400 = np.squeeze(parc_mni152.fit_transform(mitod, 'MNI152', ignore_background_data=True).T)

trc = path_data+"TRC.nii.gz"
trc_400 = np.squeeze(parc_mni152.fit_transform(trc, 'MNI152', ignore_background_data=True).T)

mrc = path_data+"MRC.nii.gz"
mrc_400 = np.squeeze(parc_mni152.fit_transform(mrc, 'MNI152', ignore_background_data=True).T)

mito_dict = {'CI': c1_400, 
             'CII': c2_400, 
             'CIV': c4_400, 
             'MitoD': mitod_400, 
             'TRC': trc_400, 
             'MRC': mrc_400}

# save parcellated mito maps
with open(path_result+'mitomaps_dict.pickle', 'wb') as f:
    pickle.dump(mito_dict, f)


################
# run analysis
# load mito maps
with open(path_result+'mitomaps_dict.pickle', 'rb') as f:
    mito_dict = pickle.load(f)

mito_df = pd.DataFrame(mito_dict)

# plot brain
for key, value in mito_dict.items():
    plot_schaefer_fsaverage(value, cmap=cmaps.matter_r,)
    plt.title(key)
    plt.savefig(path_fig+key+'.svg', dpi=600)
    plt.show()


# correlation with energy maps
energy_main = ['tca', 'oxphos', 'complex1', 'complex2', 'complex4']
energy_mean_df = pd.DataFrame(energy_mean).reset_index(drop=True)
energy_mean_df = energy_mean_df[energy_main]

# compare gene based complex maps with mitomaps
corr, _, pspin = corr_spin_test(energy_mean_df['complex1'],
                                mito_df['CI'], 
                                spins10k,
                                plot=True)
plt.savefig(path_fig+'mitomaps_complex1_corr.svg')
plt.show()

corr, _, pspin = corr_spin_test(energy_mean_df['complex2'],
                                mito_df['CII'], 
                                spins10k,
                                plot=True)
plt.savefig(path_fig+'mitomaps_complex2_corr.svg')
plt.show()

corr, _, pspin = corr_spin_test(energy_mean_df['complex4'], 
                                mito_df['CIV'], 
                                spins10k,
                                plot=True)
plt.savefig(path_fig+'mitomaps_complex4_corr.svg')
plt.show()


for map in ['MitoD', 'TRC', 'MRC']:
    corr, _, pspin = corr_spin_test(energy_mean_df['tca'],
                                    mito_df[map], 
                                    spins10k,
                                    plot=True)
    plt.savefig(path_fig+map+'_tca_corr.svg')
    plt.show()

    corr, _, pspin = corr_spin_test(energy_mean_df['oxphos'],
                                    mito_df[map], 
                                    spins10k,
                                    plot=True)
    plt.savefig(path_fig+map+'_oxphos_corr.svg')
    plt.show()

corr, pspin = pair_corr_spin(energy_mean_df[['tca', 'oxphos']], 
                             mito_df[['MitoD', 'TRC', 'MRC']], spins10k)

# fdr correction for multiple testing
model_pval = multipletests(pspin.values.flatten(), method='fdr_bh')[1]
model_pval = pd.DataFrame(model_pval.reshape(pspin.shape))
model_pval.columns = pspin.columns
model_pval.index = pspin.index

'''
model_pval
            tca    oxphos
MitoD  0.253535  0.620738
TRC    0.128237  0.059694
MRC    0.060794  0.014399
'''