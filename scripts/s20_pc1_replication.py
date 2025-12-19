
'''
Replicate main analyses with pc1 of energy pathway expression

Author: Moohebat
'''
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import colormaps as cmaps
import matplotlib.pyplot as plt
from scripts.utils import pair_corr_spin, plot_heatmap
from statsmodels.stats.multitest import multipletests

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update({'font.size': 8})

# path to directories
path_data = './data/'
path_result = './results/'
path_fig = './figures/'

# loading energy pathway data
with open(path_result+'energy_expression_matrix.pickle', 'rb') as f:
    energy_exp = pickle.load(f)
with open(path_result+'energy_mean_expression.pickle', 'rb') as f:
    energy_mean = pickle.load(f)
with open(path_result+'energy_pc1_expression.pickle', 'rb') as f:
    energy_pc1 = pickle.load(f)

energy_pc1_df = pd.DataFrame(energy_pc1)
main_energy = ['glycolysis', 'ppp', 'tca', 'oxphos', 'lactate']
energy_pc1_df = energy_pc1_df[main_energy]

# loading schaefer400 spins
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

# wagstyl genesets, layers and cells
with open(path_result + 'wag_genesets_df.pickle', 'rb') as f:
    cell_layer_df = pickle.load(f)


# energy pc1 map-map correlations
energy_pc1_corr, energy_pc1_pspin = pair_corr_spin(energy_pc1_df, energy_pc1_df, spins10k)

'''
energy_pc1_corr
            glycolysis       ppp       tca    oxphos   lactate
glycolysis    1.000000  0.219976  0.782694  0.898219  0.151334
ppp           0.219976  1.000000  0.582523  0.253461  0.829033
tca           0.782694  0.582523  1.000000  0.786169  0.587524
oxphos        0.898219  0.253461  0.786169  1.000000  0.201072
lactate       0.151334  0.829033  0.587524  0.201072  1.000000

energy_pc1_pspin
            glycolysis       ppp       tca    oxphos   lactate
glycolysis    0.000100  0.517948  0.000100  0.000100  0.736226
ppp           0.518048  0.000100  0.001100  0.469353  0.000100
tca           0.000100  0.000700  0.000100  0.000100  0.022598
oxphos        0.000100  0.471253  0.000100  0.000100  0.662234
lactate       0.734727  0.000100  0.040096  0.654735  0.000100
'''

plt.figure(figsize=(4,4))
sns.heatmap(energy_pc1_corr, annot=True, square=True,
            mask=np.tril(np.ones(energy_pc1_df.shape[1])),
            cmap=cmaps.BlueWhiteOrangeRed, cbar=True, linewidths=.3,
            xticklabels=energy_pc1_df.columns,
            yticklabels=energy_pc1_df.columns,
            vmin=-1, vmax=1, alpha=1,
            cbar_kws={'shrink': 0.25, 
                    'ticks': [-1.00, 0.0, 1.00], 
                    'pad': 0.02,
                    'aspect': 10},)
plt.yticks(rotation=0)
plt.xticks(rotation=45)
plt.title('PC1 map-map correlation')
plt.tight_layout()
plt.savefig(path_fig+'pathway_pc1_correlation.svg')
plt.show()


# correlation with multi-scale maps
all_maps = pd.concat([pet_df, meg_df, cell_layer_df, conn_df], axis=1)
all_energy_corr, all_energy_pspin = pair_corr_spin(-energy_pc1_df, all_maps, spins10k)

'''
all_energy_corr
               glycolysis       ppp       tca    oxphos   lactate
cmrglc           0.162975  0.474635  0.289110  0.155579  0.521524
cmro2           -0.031685  0.526974  0.248866 -0.021354  0.627235
cbf              0.125405  0.295678  0.133659  0.117474  0.306485
gi               0.464886  0.143858  0.327296  0.427582  0.144269
meg_delta        0.087640 -0.622156 -0.371241  0.075667 -0.720938
meg_theta        0.556712 -0.175166  0.226271  0.542334 -0.212910
meg_alpha       -0.436598  0.406348  0.003508 -0.428098  0.478266
meg_beta         0.627475  0.437855  0.719290  0.648116  0.501415
meg_lowgamma     0.466706 -0.305990  0.058373  0.452458 -0.385615
meg_highgamma    0.029843 -0.617331 -0.376584  0.011994 -0.716736
Layer 1          0.299122 -0.523832 -0.117235  0.292297 -0.532966
Layer 2          0.108505 -0.505655 -0.260108  0.007820 -0.620642
Layer 3          0.407541  0.521612  0.536338  0.381689  0.547918
Layer 4          0.493338  0.758439  0.752639  0.543145  0.832054
Layer 5          0.761685  0.077889  0.470328  0.697151 -0.049158
Layer 6         -0.103892 -0.130621 -0.156933 -0.090956 -0.121701
Cell Ex          0.503633 -0.079248  0.191002  0.428999 -0.215853
Cell In          0.398069  0.348819  0.322635  0.363978  0.244491
Cell Ast         0.304432 -0.462047 -0.110219  0.213759 -0.517141
Cell End         0.309897  0.252892  0.255136  0.316422  0.347076
Cell Mic         0.349943 -0.114553  0.159110  0.471598 -0.126843
Cell OPC        -0.202420 -0.480279 -0.532127 -0.241726 -0.583566
Cell Oli         0.068822  0.118481  0.167173  0.063504  0.217783
sc_degree       -0.066442 -0.249536 -0.293896 -0.070704 -0.317932
fc_strength     -0.128902  0.476483  0.237291 -0.091166  0.585406

all_energy_pspin
               glycolysis       ppp       tca    oxphos   lactate
cmrglc           0.561844  0.001300  0.178782  0.594541  0.002100
cmro2            0.815218  0.000200  0.258974  0.863714  0.000100
cbf              0.490051  0.056294  0.473953  0.519548  0.102690
gi               0.018598  0.627137  0.258374  0.072393  0.650335
meg_delta        0.792621  0.020398  0.469353  0.820118  0.026997
meg_theta        0.037996  0.686131  0.588341  0.065893  0.672433
meg_alpha        0.209979  0.347265  0.977102  0.250575  0.353765
meg_beta         0.017298  0.223578  0.010699  0.023298  0.234877
meg_lowgamma     0.137986  0.478352  0.865413  0.184482  0.455354
meg_highgamma    0.905809  0.000800  0.410059  0.937706  0.000600
Layer 1          0.221578  0.002400  0.789921  0.256374  0.030097
Layer 2          0.692231  0.001200  0.267173  0.987701  0.000300
Layer 3          0.034697  0.000100  0.001300  0.113589  0.000400
Layer 4          0.071393  0.000100  0.000100  0.055994  0.000100
Layer 5          0.000100  0.840316  0.011799  0.000100  0.824618
Layer 6          0.542646  0.275172  0.292671  0.658334  0.330267
Cell Ex          0.000100  0.634437  0.410359  0.001000  0.298270
Cell In          0.000100  0.000100  0.000300  0.000100  0.021198
Cell Ast         0.122688  0.001900  0.794921  0.335766  0.005999
Cell End         0.023698  0.114289  0.153085  0.030697  0.044596
Cell Mic         0.057894  0.616638  0.582742  0.000600  0.610539
Cell OPC         0.547145  0.001300  0.027697  0.469853  0.000200
Cell Oli         0.603540  0.373963  0.221078  0.678732  0.125487
sc_degree        0.814419  0.243176  0.204280  0.821718  0.200380
fc_strength      0.555044  0.013699  0.556244  0.650235  0.005599
'''

# fdr correction for multiple testing
model_pval = multipletests(all_energy_pspin.values.flatten(), method='fdr_bh')[1]
model_pval = pd.DataFrame(model_pval.reshape(25,5))
model_pval.columns = all_energy_pspin.columns
model_pval.index  = all_energy_pspin.index

'''
model_pval
               glycolysis       ppp       tca    oxphos   lactate
cmrglc           0.739268  0.006770  0.378776  0.758343  0.010095
cmro2            0.873536  0.001923  0.449608  0.894022  0.001136
cbf              0.688274  0.149719  0.679478  0.721595  0.246850
gi               0.068376  0.768551  0.449608  0.177433  0.774208
meg_delta        0.873536  0.072850  0.679478  0.873536  0.086530
meg_theta        0.107944  0.786626  0.758172  0.168095  0.785552
meg_alpha        0.416625  0.549470  0.984982  0.449608  0.552757
meg_beta         0.065524  0.423443  0.044579  0.077953  0.438202
meg_lowgamma     0.302601  0.679478  0.894022  0.384337  0.677611
meg_highgamma    0.928083  0.005263  0.618010  0.952953  0.004166
Layer 1          0.423443  0.011110  0.873536  0.449608  0.091360
Layer 2          0.786626  0.006770  0.457488  0.987701  0.002500
Layer 3          0.100862  0.001136  0.006770  0.264557  0.003125
Layer 4          0.177433  0.001136  0.001136  0.149719  0.001136
Layer 5          0.001136  0.882685  0.047576  0.001136  0.873536
Layer 6          0.739268  0.464818  0.487785  0.776337  0.536148
Cell Ex          0.001136  0.769947  0.618010  0.006249  0.490576
Cell In          0.001136  0.001136  0.002500  0.001136  0.073604
Cell Ast         0.278836  0.009499  0.873536  0.538087  0.025859
Cell End         0.077953  0.264557  0.329924  0.091360  0.123877
Cell Mic         0.150766  0.763166  0.758172  0.004166  0.763166
Cell OPC         0.739268  0.006770  0.086554  0.679478  0.001923
Cell Oli         0.762045  0.577103  0.423443  0.785570  0.280106
sc_degree        0.873536  0.447014  0.411854  0.873536  0.410615
fc_strength      0.739268  0.053510  0.739268  0.774208  0.024998
'''

# plot
plt.figure(figsize=(5,2.5))
plot_heatmap(all_energy_corr, model_pval)
plt.savefig(path_fig+'pc1_all_corr_heatmap_wagstyl_asteriks.svg')
plt.show()
