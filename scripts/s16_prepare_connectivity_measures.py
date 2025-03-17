
'''
calculating sc and fc connectivity measures
Author: Moohebat
Date: 28/06/2024
'''
# import packages
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, zscore
from nilearn.datasets import fetch_atlas_schaefer_2018
from bct import degree, centrality, distance, clustering
from netneurotools import datasets, utils, plotting
from scipy.spatial.distance import squareform, pdist
from scripts.utils import plot_schaefer_fsaverage

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update({'font.size': 8})

# path to directories
path_data = './data/'
path_result = './results/'
path_fig = './figures/'

# functional networks for participation coeff
schaefer400 = fetch_atlas_schaefer_2018()
rsn_mapping = []
for row in range(len(schaefer400['labels'])):
    rsn_mapping.append(schaefer400['labels'][row].decode('utf-8').split('_')[2])
rsn_mapping = np.array(rsn_mapping)

############
# binary sc
sc = np.load(path_data+'consensusSC.npy')

# distance
path = path_data+'schaefer_coords/'
coords = np.genfromtxt(path+'Schaefer_400_centres.txt')[:, 1:]
eu_distance = squareform(pdist(coords, metric='euclidean'))
eu = np.mean(eu_distance, axis=1).reshape(-1, 1)


sc_deg = degree.degrees_und(sc).reshape(-1, 1)
sc_between = centrality.betweenness_bin(sc).reshape(-1, 1)
sc_cluster = clustering.clustering_coef_bu(sc).reshape(-1, 1)
sc_dist = distance.distance_bin(sc)
sc_spath = np.mean(sc_dist, axis=1).reshape(-1, 1)
sc_effic = distance.efficiency_bin(sc, local=True,).reshape(-1, 1)
sc_partic = centrality.participation_coef(sc, rsn_mapping, 
                                          'undirected').reshape(-1, 1)

# sc measure dataframe
sc_attr = np.concatenate((eu, sc_deg, sc_between, sc_cluster,
                          sc_spath, sc_effic, sc_partic), axis=1)

sc_df = pd.DataFrame(sc_attr, columns=['euc_dist', 'sc_degree', 
                                        'sc_betweenness', 'sc_clustering', 
                                        'sc_spath', 'sc_efficiency', 
                                        'sc_participation'])

#####################
# fc network measure
fc = np.load(path_data+'haemodynamic_connectivity.npy')
# strength is the sum of weights of edges connected to the node
fc_strength = degree.strengths_und(abs(fc)).reshape(-1, 1)
fc_between = centrality.betweenness_wei(abs(1/fc)).reshape(-1, 1)
fc_cluster = clustering.clustering_coef_wu(abs(fc)).reshape(-1, 1)
fc_partic = centrality.participation_coef(abs(fc), rsn_mapping).reshape(-1, 1)
fc_dist, _ = distance.distance_wei(abs(1/fc))
fc_spath = np.mean(fc_dist, axis=1).reshape(-1, 1)

fc_strength_pos, _, _, _ = degree.strengths_und_sign(fc)
fc_partic_pos, _ = centrality.participation_coef_sign(fc, rsn_mapping)

# fc dataframe
fc_attr = np.concatenate((fc_strength, fc_between, 
                          fc_cluster, fc_partic, fc_spath,
                          fc_strength_pos.reshape(-1,1),
                          fc_partic_pos.reshape(-1,1)), axis=1)
fc_df = pd.DataFrame(fc_attr, columns=['fc_strength', 'fc_betweenness', 
                                            'fc_clustering', 'fc_participation',
                                            'fc_spath', 'fc_strength_pos',
                                            'fc_partic_pos'])