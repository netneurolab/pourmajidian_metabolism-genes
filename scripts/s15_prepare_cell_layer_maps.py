
'''
Prepare layer- and cell-specific marker genes
from wagstyl 2023 cartography paper and 
make expression maps for each cell-type

Author: Moohebat

'''

import numpy as np
import pandas as pd
import pickle
import abagen
import matplotlib.pyplot as plt
import colormaps as cmaps
from scripts.utils import geneset_expression, plot_schaefer_fsaverage

# path to directories
path_data = './data/'
path_result = './results/'
path_fig = './figures/'

# load wagstyl genesets
wag_genesets = pd.read_csv(path_data+'wagstyl_genesets.csv')

# keep cell and layer markers
to_keep = ['gene.symbol', 'Layer 1','Layer 2', 'Layer 3', 'Layer 4', 
           'Layer 5', 'Layer 6', 'Cell Ex', 'Cell In', 
           'Cell Ast', 'Cell End', 'Cell Mic', 'Cell OPC', 'Cell Oli']
genesets = wag_genesets[to_keep]

# make cell and layer gene sets
geneset_dict = {}
for col in genesets.columns[1:]:
    geneset_dict[col] = genesets['gene.symbol'][genesets[col]==True].reset_index(drop=True)

# save
with open(path_result+'wag_cell_layer_genes.pickle', 'wb') as f:
    pickle.dump(geneset_dict, f)

# load ahba expression matrix 
with open(path_data + 'expression_ds01.pickle', 'rb') as f:
    expression_ds01 = pickle.load(f)

# get gene set expressions
geneset_exp = {}
geneset_mean = {}

for key, value in geneset_dict.items():
    geneset_exp[key] = geneset_expression(expression_ds01, value, key, path_result)
    geneset_mean[key] = np.mean(geneset_exp[key], axis=1)

# convert to dataframe
geneset_mean_df = pd.DataFrame(geneset_mean).reset_index(drop=True)

# plot
for key, value in geneset_mean.items():
    plot_schaefer_fsaverage(value, cmap=cmaps.matter_r)
    plt.title(key)
    plt.show()