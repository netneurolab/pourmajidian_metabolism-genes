
'''
Retrieve and parcellate PET, MEG and FC gradient
maps from neuromaps

Author: Moohebat
Date: 26/06/2024
'''

import pickle
import numpy as np
import pandas as pd
from netneurotools import datasets as nntdata
from neuromaps.datasets.contributions import available_annotations
from neuromaps.datasets import fetch_annotation
from neuromaps.parcellate import Parcellater
from neuromaps.images import dlabel_to_gifti
from neuromaps.transforms import fslr_to_fslr

# path to directories
path_data = './data/'
path_result = './results/'
path_fig = './figures/'

########################
# PET metbaolic maps
# cmro2, cmrglu, cbf, cbv
target = ['cmruglu', 'cmr02', 'cbf', 'cbv']
#getting schaefer400 in flr32k dlabel file
schaefer32k = nntdata.fetch_schaefer2018('fslr32k')['400Parcels7Networks']
parc32k = Parcellater(dlabel_to_gifti(schaefer32k), 'fsLR')

pet_dict = {}
for i in range(len(target)):
    surface = fetch_annotation(desc=target[i])  # glucose metab. this is in fslr 164k.
    surface32k = fslr_to_fslr(surface, target_density='32k')
    pet_dict[target[i]] = parc32k.fit_transform(surface32k, 'fsLR', ignore_background_data=True)
    np.save(path_data+target[i], map)

# glycolytic index
pet_dict['gi'] = np.load(path_data+'gi.npy')

# convert to dataframe
pet_df = pd.DataFrame(pet_dict)
pet_df.columns = ['cmrglc', 'cmro2', 'cbf', 'cbv', 'gi']

# save
with open(path_data + 'pet_df.pickle', 'wb') as f:
    pickle.dump(pet_df, f)


##########
# MEG maps
target = ['megalpha', 'megbeta', 'megdelta', 'meggamma1', 'meggamma2', 'megtheta', 'megtimescale']
schaefer32k = nntdata.fetch_schaefer2018('fslr32k')['400Parcels7Networks']
schaefer32kgii = dlabel_to_gifti(schaefer32k)
schaefer4k = fslr_to_fslr(schaefer32kgii, target_density='4k', method='nearest')
parc4k = Parcellater(schaefer4k, 'fsLR')

meg_dict = {}
for i in range(len(target)):
    surface = fetch_annotation(desc=target[i])  #MEG maps are in fSLr 4k
    meg_dict[target[i]] = parc4k.fit_transform(surface, 'fsLR', ignore_background_data=True)
    np.save(path_data+target[i], map)

# convert to dataframe
meg_df = pd.DataFrame(meg_dict)
meg_df.columns = ['meg_alpha', 'meg_beta', 'meg_delta', 
                  'meg_lowgamma', 'meg_highgamma', 'meg_theta',
                  'meg_timescale']

# save
with open(path_data + 'meg_df.pickle', 'wb') as f:
    pickle.dump(meg_df, f)

############################
#fc gradients from margulies
target = ['fcgradient01', 'fcgradient02', 'fcgradient03']
schaefer32k = nntdata.fetch_schaefer2018('fslr32k')['400Parcels7Networks']
parc32k = Parcellater(dlabel_to_gifti(schaefer32k), 'fsLR')

fc_dict = {}
for i in range(len(target)):
    surface = fetch_annotation(desc=target[i]) #gifti fsLR32k
    fc_dict[target[i]] = parc32k.fit_transform(surface, 'fsLR', ignore_background_data=True)
    np.save(path_data+target[i], map)

# convert to dataframe
fc_df = pd.DataFrame(fc_dict)
fc_df.columns = ['fc1', 'fc2', 'fc3']

# save
with open(path_data + 'fc_df.pickle', 'wb') as f:
    pickle.dump(fc_df, f)


#myelin map
available_annotations(desc = 'myelinmap')  #fsLR, 32k

schaefer32k = nntdata.fetch_schaefer2018('fslr32k')['400Parcels7Networks']
parc32k = Parcellater(dlabel_to_gifti(schaefer32k), 'fsLR')
myelin = fetch_annotation(desc='myelinmap') #gifti fsLR32k
myelin_map = parc32k.fit_transform(myelin, 'fsLR', ignore_background_data=True) #(400,) array

np.save(path_data+'myelin', myelin_map)
