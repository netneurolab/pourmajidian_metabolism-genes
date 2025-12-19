
'''
Correlation of energy maps with functional connectivity gradients

Author: Moohebat
Date: 20/06/2024
'''

# fc gradients are from margulies et al 2016
# maps are parcellated in neuromaps scripts

# import packages
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scripts.utils import corr_spin_test, plot_schaefer_fsaverage
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update({'font.size': 8})

# path to directories
path_data = './data/'
path_result = './results/'
path_fig = './figures/'

# load energy pathway data
with open(path_result+'energy_mean_expression.pickle', 'rb') as f:
    energy_mean = pickle.load(f)

# load spins
spins10k = np.load(path_data+'spins10k.npy')

# load functional connectivity gradient maps
fc1_map = np.load(path_data+'fc1_margulies.npy')
fc2_map = np.load(path_data+'fc2_margulies.npy')
fc3_map = np.load(path_data+'fc3_margulies.npy')

# focusing on main energy pathways
energy_main =['glycolysis', 'ppp', 'tca', 'oxphos', 'lactate']
energy_mean = {key: energy_mean[key] for key in energy_main}

cmap = LinearSegmentedColormap.from_list('custom_cmap', ['lightskyblue', 'white', 'coral'])

# plotting fc gradients
fcs = [fc1_map, fc2_map, fc3_map]
for i, fc in enumerate(fcs):
    plot_schaefer_fsaverage(-fc, cmap = cmap)
    plt.title('FC Gradient'+str(i+1))
    # plt.savefig(path_fig+'fc_invert'+str(i)+'.svg')
    plt.show()

# correlations with energy mean maps
fc1 = pd.DataFrame(index=np.arange(5), columns=['pathway', 'r', 'pspin'])
fc2 = pd.DataFrame(index=np.arange(5), columns=['pathway', 'r', 'pspin'])
fc3 = pd.DataFrame(index=np.arange(5), columns=['pathway', 'r', 'pspin'])
fc_dict = {'fc1': fc1, 'fc2': fc2, 'fc3': fc3}

margul = np.array(fcs).T
for j, (k, v) in enumerate(fc_dict.items()):
    print(j,k)
    for i, (key, value) in enumerate(energy_mean.items()):
        v['pathway'][i] = key
        v['r'][i], _, v['pspin'][i] = corr_spin_test(np.array(energy_mean[key]), 
                                                                       -margul[:, j], spins10k)

'''
fc1
      pathway         r     pspin
0  glycolysis  0.083756  0.720328
1         ppp  0.466072    0.0005
2         tca  0.369101  0.030297
3      oxphos  0.112265  0.583842
4     lactate  0.286454  0.145185
'''

'''
fc2
      pathway         r     pspin
0  glycolysis  0.456285  0.015498
1         ppp -0.097568  0.630037
2         tca  0.271549  0.371363
3      oxphos  0.456028  0.010399
4     lactate  0.325808  0.286171
'''

'''
fc3
      pathway         r     pspin
0  glycolysis -0.151564  0.162284
1         ppp  0.046242  0.605139
2         tca -0.067337  0.633837
3      oxphos -0.072347  0.548245
4     lactate -0.148383  0.181282
'''

# plotting
fig, axs = plt.subplots(1, len(fc_dict), figsize=(1.3*len(fc_dict), 1.5))
for ax, (key, value) in zip(axs, fc_dict.items()):
    colors = ['coral' if pspin<0.05 else 'lightgrey' for pspin in value['pspin'][::-1]]
    ax.barh(y=value['pathway'][::-1], width=value['r'][::-1], color=colors, height=0.5, alpha=0.8)
    ax.set_xlim(-0.7,0.7)
    ax.set_title(key)
    ax.set_xlabel('rho')
    ax.set_yticks(value['pathway'][::-1])
    sns.despine()

for i, ax in enumerate(axs):
    if i != 0:
        ax.set_yticklabels(['']*len(ax.get_yticks()))
    if i == 0:    
        ax.set_ylabel('mean expression')
plt.tight_layout()
plt.savefig(path_fig+'fc_correlations.svg')
plt.show()

