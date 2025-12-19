
'''
Make energy pathway gene expression maps 
in Schaefer-400 parcellation

1. Pathway gene expression matrices
2. Pathway mean gene expression maps
3. PC1 of pathway gene expression marix

Author: Moohebat
Date: 19/06/2024
'''

import numpy as np
import pandas as pd
import pickle
import abagen
import matplotlib.pyplot as plt
import colormaps as cmaps
from sklearn.decomposition import PCA
from scripts.utils import (load_expression, 
                           filter_expression_ds, 
                           geneset_expression)

# path to directories
path_data = './data/'
path_result = './results/'
path_fig = './figures/'
path_all= './results/energy_sets/all_pathways/'

# retrieve expression data for schaefer400 parcellation
expression_schaefer400 = load_expression(scale=400)

# save expression dict for filtering
with open(path_data + 'expression_dict_schaefer400.pickle', 'wb') as f:
    pickle.dump(expression_schaefer400, f)

# filter genes with differential stability >0.1
expression_ds01, ds = filter_expression_ds(expression_schaefer400, ds=0.1)
# dataframe of 400 x 8687

diff_stable = {'gene': np.array(expression_schaefer400['15496'].columns), 'ds': ds}
diff_stable = pd.DataFrame(diff_stable)
diff_stable.to_csv(path_result+'gene_ds_all.csv')

# save
path_data = 'D:/McGill/Dagher_lab/my_neuro_project/project_data/'
with open(path_data + 'expression_ds01.pickle', 'wb') as f:
    pickle.dump(expression_ds01, f)


##############
# run analysis

# load expression dictionary
with open(path_data + 'expression_dict_schaefer400.pickle', 'rb') as f:
    expression_schaefer400 = pickle.load(f)

with open(path_data + 'expression_ds01.pickle', 'rb') as f:
    expression_ds01 = pickle.load(f)

##########################
# prepare energy gene sets
# from the glycolysis list remove HK2 gene
# because it's entry to both ppp and glycolysis

all_pathways = ['glycolysis', 'ppp', 'tca', 'oxphos', 'lactate', 
                'kb_util','fa_metabolism','glycogen_metabolism',
                'complex1', 'complex2', 'complex3', 'complex4',
                'atpsynth', 'ros_detox', 'ros_gen', 'no_signalling',
                'atpase', 'pdc', 'bcaa_cat']

# load energy gene set csv files
energy_dict = {}
for item in all_pathways:
    glist = pd.read_csv(path_all + item + '.csv', header=0).dropna()
    energy_dict[item] = glist.values.flatten()

energy_dict['kb_util'] = np.append(energy_dict['kb_util'], ['BDH1', 'BDH2'])

# adding new energy related gene sets
# malate-aspartate shuttle, 'R-HSA-9856872'
energy_dict['mas'] = ['SLC25A18', 'SLC25A22', 'SLC25A13', 'SLC25A12', 'SLC25A11'
                      'MDH1', 'MDH2', 'MDH1B',
                      'GOT1', 'GOT2']

# glycerol-phosphate shuttle, 'R-HSA-188467'
energy_dict['gps'] = ['GPD1', 'GPD2', 'GPD1L']

# creatine kinase activity, GO:0004111
energy_dict['creatine_kinase'] = ['CKB', 'CKM','CKMT1A', 
                           'CKMT1B', 'CKMT2']

# glutamate-glutamine A-N cycle
energy_dict['gln_glu_cycle'] = ['GLUL', 'GLS', 'GLS2', 
                                'SLC1A3', 'SLC1A2', 'SLC1A1', 'SLC1A6', 'SLC1A7', 
                                'SLC38A1', 'SLC38A2', 'SLC38A3', 'SLC38A4', 
                                'SLC38A5', 'SLC38A6', 'SLC38A7']

# save gene list dictionary
with open(path_result + 'energy_genelist_dict.pickle', 'wb') as f:
    pickle.dump(energy_dict, f)


#######################################
# retrieving energy expression matrices

# load gene set
with open(path_result + 'energy_genelist_dict.pickle', 'rb') as f:
    energy_dict = pickle.load(f)

energy_exp = {}
energy_mean = {}
energy_pc1 = {}

pca = PCA(n_components=1)

# getting expression matrices, pc1 and mean maps for energy pathways
for key, value in energy_dict.items():
    energy_exp[key] = geneset_expression(expression_ds01, value, key, path_result)
    energy_mean[key] = np.mean(energy_exp[key], axis=1)
    energy_pc1[key] = np.squeeze(pca.fit_transform(energy_exp[key]))

# save
with open(path_result + 'energy_expression_matrix.pickle', 'wb') as f:
    pickle.dump(energy_exp, f)
with open(path_result + 'energy_mean_expression.pickle', 'wb') as f:
    pickle.dump(energy_mean, f)
with open(path_result + 'energy_pc1_expression.pickle', 'wb') as f:
    pickle.dump(energy_pc1, f)
