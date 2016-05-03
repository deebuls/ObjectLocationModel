import pymc
import numpy as np

import pandas as pd
import numpy as np
aruba_dataset_path = '/data/dataDeebul/thesis/dataset/strands/aruba/locations.names'

location_names = {}
count = 0
with open(aruba_dataset_path, 'r') as content_file:
    content = content_file.read()
    content = str.splitlines(content)
for count,location in enumerate(content):
    location_names[count] = location
print(location_names)


aruba_dataset_path = '/data/dataDeebul/thesis/dataset/strands/aruba/locations.min'

dataset = pd.read_csv(aruba_dataset_path, names=['location', 'time'])
dataset['time'] = dataset.index
dataset['time'] = pd.to_timedelta(dataset['time'], unit='m')
dataset['minute'] = (dataset['time']/ np.timedelta64(1, 'm')).astype(int)


#Renaming location number with their names
for key,value in location_names.items():
    dataset.ix[dataset.location ==key, 'location_name'] = value
    
def mod_for_minute(row):
    return row['minute'] % 1440
dataset['sep_minute'] = dataset.apply(mod_for_minute, axis=1)

living_room_data = dataset[dataset['location_name'].isin(['Living room'])]
living_room_data.loc[:,'circular_minute'] = ((living_room_data.loc[:,'sep_minute']/1440) * 2 * np.pi ) - np.pi


#Model
mu = pymc.Uniform('mu', lower=-np.pi, upper=np.pi)
kappa = pymc.Uniform('kappa', lower=0.0, upper=100.0)
y = pymc.VonMises('y',mu, kappa, value=living_room_data['circular_minute'].values, observed=True)