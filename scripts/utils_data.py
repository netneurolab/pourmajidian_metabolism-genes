
'''
script to prepare spins for spin tests
and prepare yeo, von economo and mesulam network labels

date: 17/09/2024
'''

import numpy as np
import pandas as pd
from netneurotools import stats
from nilearn.datasets import fetch_atlas_schaefer_2018

# path to directories
path_data = './data/'
path_result = './results/'
path_fig = './figures/'

##################################
# creating spins for schaefer 400
path = 'D:/McGill/Dagher_lab/Neuroenergetics_project/from_Justine/coordinates/'
coords = np.genfromtxt(path+'Schaefer_400_centres.txt')[:, 1:] #centroid coordinates
nnodes = coords.shape[0]
hemiid = np.zeros((nnodes,))
hemiid[:int(nnodes/2)] = 1

nspins1k = 1000
spins1k = stats.gen_spinsamples(coords, hemiid, n_rotate=nspins1k, seed=42, method='vasa')

nspins5k = 5000
spins5k = stats.gen_spinsamples(coords, hemiid, n_rotate=nspins5k, seed=42, method='vasa')

nspins10k = 10000
spins10k = stats.gen_spinsamples(coords, hemiid, n_rotate=nspins10k, seed=42, method='vasa')

#####################################
#creating spins for for schaefer 100
path = 'D:/McGill/Dagher_lab/Neuroenergetics_project/from_Justine/coordinates/'
coords = np.genfromtxt(path+'Schaefer_100_centres.txt')[:, 1:]
nnodes = coords.shape[0]
hemiid = np.zeros((nnodes,))
hemiid[:int(nnodes/2)] = 1

nspins1k = 1000
spins1k_schaefer100 = stats.gen_spinsamples(coords, hemiid, n_rotate=nspins1k, seed=42, method='vasa')
np.save(path_data+'spins1k_schaefer100.npy', spins1k_schaefer100)

nspins10k = 10000
spins10k_schaefer100 = stats.gen_spinsamples(coords, hemiid, n_rotate=nspins10k, seed=42, method='vasa')
np.save(path_data+'spins10k_schaefer100.npy', spins10k_schaefer100)


'''
yeo networks, von economo and mesulam classes
'''

##############
# schaefer400 

##############
# yeo mapping
schaefer400 = fetch_atlas_schaefer_2018()

yeo_schaefer400 = schaefer400
for i, label in enumerate(schaefer400['labels']):
    yeo_schaefer400['labels'][i] = label.decode('utf-8').split("_")[2]

schaefer400_yeo = np.array(yeo_schaefer400['labels'].astype(str))

np.save(path_data+'yeo_schaefer400.npy', schaefer400_yeo)

#####################
#von economo mapping
#    label                       name
# 0      1              primary motor
# 1      2                association
# 2      3               association2
# 3      4  primary/secondary sensory
# 4      5            primary sensory
# 5      6                     limbic
# 6      7                    insular
path = 'D:/McGill/Dagher_lab/my_neuro_project/von_economo/'
ve_schaefer400 = pd.read_csv(path+'von_economo.csv', header=None)
ve_schaefer400.columns= ['label']
ve_names = pd.read_csv(path+'von_economo_names.csv')

for i in range(ve_schaefer400['label'].shape[0]):
    for j in ve_names['label']:
        if ve_schaefer400['label'][i] == ve_names['label'][j-1]:
            ve_schaefer400['label'][i] = ve_names['name'][j-1]
schaefer400_ve = np.array(ve_schaefer400['label'].astype(str))

np.save(path_data+'ve_schaefer400.npy', schaefer400_ve)

# np.savetxt('C:/Users/moohe/Desktop/'+'ve_schaefer400.csv', schaefer400_ve)
# df = pd.DataFrame(schaefer400_ve)
# df.to_csv('C:/Users/moohe/Desktop/'+'ve_schaefer400.csv')

##################
# mesulam mapping
#mesulam classes
path = 'D:/McGill/Dagher_lab/my_neuro_project/von_economo/'
mesulam_schaefer400 = pd.read_csv(path+'mesulam_schaefer400.csv')
mesulam_names = pd.read_csv(path+'mesulam_schaefer400_names.csv')

for i in range(mesulam_schaefer400['labels'].shape[0]):
    for j in mesulam_names['label']:
        if mesulam_schaefer400['labels'][i] == mesulam_names['label'][j-1]:
            mesulam_schaefer400['labels'][i] = mesulam_names['name'][j-1]

schaefer400_mesulam = np.array(mesulam_schaefer400['labels'].astype(str))
np.save(path_data+'mesulam_schaefer400.npy', schaefer400_mesulam)
#    label        name
# 0      1  paralimbic
# 1      2  hetermodal
# 2      3    unimodal
# 3      4   idiotypic

###############
# schaefer 100

##############
# yeo mapping
schaefer100 = fetch_atlas_schaefer_2018(n_rois=100)

yeo_schaefer100 = schaefer100
for i, label in enumerate(schaefer100['labels']):
    yeo_schaefer100['labels'][i] = label.decode('utf-8').split("_")[2]

schaefer100_yeo = np.array(yeo_schaefer100['labels'].astype(str))

np.save(path_data+'yeo_schaefer100.npy', schaefer100_yeo)

#####################
#von economo mapping
#    label                       name
# 0      1              primary motor
# 1      2                association
# 2      3               association2
# 3      4  primary/secondary sensory
# 4      5            primary sensory
# 5      6                     limbic
# 6      7                    insular
path = 'D:/McGill/Dagher_lab/my_neuro_project/von_economo/'
ve_schaefer100 = pd.read_csv(path+'voneconomo_Schaefer100.csv', header=None)
ve_schaefer100.columns= ['label']
ve_names = pd.read_csv(path+'von_economo_names.csv')

for i in range(ve_schaefer100['label'].shape[0]):
    for j in ve_names['label']:
        if ve_schaefer100['label'][i] == ve_names['label'][j-1]:
            ve_schaefer100['label'][i] = ve_names['name'][j-1]
schaefer100_ve = np.array(ve_schaefer100['label'].astype(str))

np.save(path_data+'ve_schaefer100.npy', schaefer100_ve)

# np.savetxt('C:/Users/moohe/Desktop/'+'ve_schaefer100.csv', schaefer100_ve)
# df = pd.DataFrame(schaefer100_ve)
# df.to_csv('C:/Users/moohe/Desktop/'+'ve_schaefer100.csv')

#################
#mesulam mapping
path = 'D:/McGill/Dagher_lab/my_neuro_project/von_economo/'
mesulam_schaefer100 = pd.read_csv(path+'mesulam_scale100.csv', header=None)
mesulam_schaefer100.columns= ['label']
mesulam_names = pd.read_csv(path+'mesulam_schaefer400_names.csv')

for i in range(mesulam_schaefer100['label'].shape[0]):
    for j in mesulam_names['label']:
        if mesulam_schaefer100['label'][i] == mesulam_names['label'][j-1]:
            mesulam_schaefer100['label'][i] = mesulam_names['name'][j-1]

schaefer100_mesulam = np.array(mesulam_schaefer100['label'].astype(str))
np.save(path_data+'mesulam_schaefer100.npy', schaefer100_mesulam)
#    label        name
# 0      1  paralimbic
# 1      2  hetermodal
# 2      3    unimodal
# 3      4   idiotypic