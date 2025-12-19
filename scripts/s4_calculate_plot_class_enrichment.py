
'''
Distribution of energy maps across
the microstructural and functional network classes
including, yeo, von economo and mesulam

Author: Moohebat
Date: 20/06/2024
'''
# import packages
import pickle
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from netneurotools import plotting
from scripts.utils import class_enrichment

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update({'font.size': 8})

# path to directories
path_data = './data/'
path_result = './results/'
path_fig = './figures/'

###########
# load data
# loading energy pathway data
with open(path_result+'energy_mean_expression.pickle', 'rb') as f:
    energy_mean = pickle.load(f)

# loading class labels
yeo_schaefer400 = np.load(path_data+'yeo_schaefer400.npy')
ve_schaefer400 = np.load(path_data+'ve_schaefer400.npy', allow_pickle=True)
mesulam_schaefer400 = np.load(path_data+'mesulam_schaefer400.npy', allow_pickle=True)

# loading spins
spins10k = np.load(path_data+'spins10k.npy')

#####################
# convert label names
# Mapping dictionary
mapping = {
    'Cont': 'FP',
    'Default': 'DM',
    'DorsAttn': 'DA',
    'Limbic': 'Lim',
    'SalVentAttn': 'SA',
    'SomMot': 'SM',
    'Vis': 'Vis'
}
# Convert the values using the mapping dictionary
yeo_schaefer400 = np.array([mapping[val] for val in yeo_schaefer400])

## ve
mapping = {
    'association':'Ac',
    'association2':'Ac2',
    'insular': 'Ins',
    'limbic': 'Lim',
    'primary motor': 'PM',
    'primary sensory': 'PS',
    'primary/secondary sensory': 'PSS'
}
# Convert the values using the mapping dictionary
ve_schaefer400 = np.array([mapping[val] for val in ve_schaefer400])

## mesulam
mapping = {
    'hetermodal': 'HM', 
    'idiotypic': 'ID', 
    'paralimbic': 'PLB', 
    'unimodal': 'UM'
}
# Convert the values using the mapping dictionary
mesulam_schaefer400 = np.array([mapping[val] for val in mesulam_schaefer400])

# order based on hierarchy level for plots
classes = {'yeo':{
    'order': ['SM','Vis', 'DA', 'SA', 'FP','DM','Lim'], 
    'map': yeo_schaefer400},
    've': {
        'order': ['PM','PS','PSS', 'Ac','Ac2','Ins','Lim'],
        'map': ve_schaefer400},
    'mesulam': {
        'order': ['ID', 'UM', 'HM', 'PLB'],
        'map': mesulam_schaefer400}}


##############
# run analysis

# again focusing on main energy pathways
energy_main =['glycolysis', 'ppp', 'tca', 'oxphos', 'lactate']
energy_mean = {key: value for key, value in energy_mean.items() if key in energy_main}


# class enrichment for each network separate
energy_mean_df = pd.DataFrame.from_dict(energy_mean, 
                                        orient='columns').reset_index(drop=True)
main_energy = ['glycolysis', 'ppp', 'tca', 'oxphos', 'lactate']
energy_mean_df = energy_mean_df[main_energy]

#yeo
class_enrichment(energy_mean_df, 
                 yeo_schaefer400, 
                 spins10k,
                 classes['yeo']['order'],
                 path_fig,
                 'yeo_enrichment')


'''
order: ['SM','Vis', 'DA', 'SA', 'FP','DM','Lim']
pspins
{'glycolysis': 0.048795120487951205, 'ppp': 0.20437956204379562, 'tca': 0.0010998900109989002, 'oxphos': 0.030996900309969003, 'lactate': 0.07389261073892611}
{'glycolysis': 0.0030996900309969004, 'ppp': 0.0953904609539046, 'tca': 0.7542245775422458, 'oxphos': 0.0038996100389961006, 'lactate': 0.27747225277472254}
{'glycolysis': 0.46715328467153283, 'ppp': 0.46905309469053097, 'tca': 0.2515748425157484, 'oxphos': 0.5485451454854514, 'lactate': 0.4336566343365663}
{'glycolysis': 0.6678332166783322, 'ppp': 0.050794920507949204, 'tca': 0.11278872112788721, 'oxphos': 0.7777222277772223, 'lactate': 0.6432356764323568}
{'glycolysis': 0.20977902209779023, 'ppp': 0.6472352764723528, 'tca': 0.7931206879312068, 'oxphos': 0.42195780421957807, 'lactate': 0.09269073092690731}
{'glycolysis': 0.9636036396360363, 'ppp': 0.060693930606939304, 'tca': 0.0292970702929707, 'oxphos': 0.7756224377562244, 'lactate': 0.9141085891410858}
{'glycolysis': 0.6360363963603639, 'ppp': 0.006199380061993801, 'tca': 0.2826717328267173, 'oxphos': 0.8123187681231877, 'lactate': 0.11858814118588142}
'''
#von economo
class_enrichment(energy_mean_df, 
                 ve_schaefer400, 
                 spins10k,
                 classes['ve']['order'],
                 path_fig,
                 've_enrichment')

'''
order: ['PM','PS','PSS', 'Ac','Ac2','Ins','Lim'],
pspins:
{'glycolysis': 0.0108989101089891, 'ppp': 0.34506549345065496, 'tca': 0.0010998900109989002, 'oxphos': 0.007699230076992301, 'lactate': 0.006999300069993001}
{'glycolysis': 0.1606839316068393, 'ppp': 9.999000099990002e-05, 'tca': 0.10078992100789921, 'oxphos': 0.402959704029597, 'lactate': 0.8033196680331967}
{'glycolysis': 0.06989301069893011, 'ppp': 0.20407959204079593, 'tca': 0.4296570342965703, 'oxphos': 0.09609039096090391, 'lactate': 0.41255874412558746}
{'glycolysis': 0.22647735226477353, 'ppp': 0.18248175182481752, 'tca': 0.5090490950904909, 'oxphos': 0.38536146385361464, 'lactate': 0.37236276372362764}
{'glycolysis': 0.6725327467253275, 'ppp': 0.6161383861613838, 'tca': 0.36496350364963503, 'oxphos': 0.7004299570042996, 'lactate': 0.7186281371862814}
{'glycolysis': 0.07039296070392961, 'ppp': 0.0033996600339966003, 'tca': 0.00019998000199980003, 'oxphos': 0.17698230176982302, 'lactate': 0.0198980101989801}
{'glycolysis': 0.7947205279472053, 'ppp': 0.061993800619938005, 'tca': 0.33486651334866513, 'oxphos': 0.7011298870112989, 'lactate': 0.29847015298470153}
'''
#mesulam
class_enrichment(energy_mean_df, 
                 mesulam_schaefer400, 
                 spins10k,
                 classes['mesulam']['order'],
                 path_fig,
                 'mesulam_enrichment')

'''
order: ['ID', 'UM', 'HM', 'PLB'],
pspins:
{'glycolysis': 0.31926807319268075, 'ppp': 0.00019998000199980003, 'tca': 0.00039996000399960006, 'oxphos': 0.36826317368263173, 'lactate': 0.0150984901509849}
{'glycolysis': 0.518048195180482, 'ppp': 0.8448155184481552, 'tca': 0.8473152684731526, 'oxphos': 0.7352264773522648, 'lactate': 0.551944805519448}
{'glycolysis': 0.3103689631036896, 'ppp': 0.7238276172382762, 'tca': 0.4370562943705629, 'oxphos': 0.9773022697730227, 'lactate': 0.17348265173482652}
{'glycolysis': 0.5534446555344466, 'ppp': 0.018098190180981903, 'tca': 0.10668933106689331, 'oxphos': 0.7558244175582441, 'lactate': 0.10108989101089891}
'''

############################################
# plotting brain point plot for each network
s_kw = {'cmap' :mpl.colors.ListedColormap(['gainsboro', 'slategray'])}

path = path_data+'data/atlases/schaefer_coords/'
coords = np.genfromtxt(path+'Schaefer_400_centres.txt')[:, 1:] # centroid coordinates

# yeo
for i, item in enumerate(np.unique(yeo_schaefer400)):
    mask = (yeo_schaefer400 == item).astype(int)
    plotting.plot_point_brain(mask, coords, views='sag', 
                              size=23, linewidth=0.1, alpha=0.8, **s_kw,
                              views_size=(3,3))
    plt.title(item)
    plt.tight_layout()
    plt.savefig(path_fig + 'yeo_' + str(i) +'.svg')
    # plt.show()

# mesulam
for i, item in enumerate(np.unique(mesulam_schaefer400)):
    mask = (mesulam_schaefer400 == item).astype(int)
    plotting.plot_point_brain(mask, coords, views='sag', 
                              size=23, linewidth=0.1, alpha=0.8, **s_kw,
                              views_size=(3,3))
    plt.title(item)
    plt.tight_layout()
    plt.savefig(path_fig + 'mesulam_' + str(i) +'.svg')
    plt.show()

#ve
for i, item in enumerate(np.unique(ve_schaefer400)):
    mask = (ve_schaefer400 == item).astype(int)
    plotting.plot_point_brain(mask, coords, views='sag', 
                              size=23, linewidth=0.1, alpha=0.8, **s_kw,
                              views_size=(3,3))
    plt.title(item)
    plt.tight_layout()
    plt.savefig(path_fig + 've_' + str(i) +'.svg')
    plt.show()