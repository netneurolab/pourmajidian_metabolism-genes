
'''
Script to prepare spins for spin tests

Author: Moohebat
Date: 17/09/2024
'''

import numpy as np
import pandas as pd
from netneurotools import stats

# path to directories
path_data = './data/'
path_result = './results/'
path_fig = './figures/'


###############################
# create spins for schaefer 400

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


##################################
#create spins for for schaefer 100
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