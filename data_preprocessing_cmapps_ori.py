# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 09:25:03 2022

@author: Marcel Henkel
"""

import pandas as pd
from sklearn import preprocessing
import numpy as np


#Make Changes: Return all max Values (not only RUL)

###Preprocessing CMAPSS Data, by dropping unused columns, normalising and calculating RUL Labelvektor###
###Returning preprocessed Data as Sequences with length sl, beginning with [0,sl-1]00 measurement, then [1,sl] measurement,
### and so on. Stopping at [-1-sl, sl] measurment of an engine and continue with the next engine, RUL Label Vector and Max RUL Value###  
sl=50
path='Data/CMAPSS/train_FD001.txt'
rul_path = 'Data/CMAPSS/RUL_FD001.txt'
short=False

sequence_length = sl
train_df = pd.read_csv(path, sep=" ", header=None)
#print(train_df)
train_df.drop(train_df.columns[[26, 27]], axis=1, inplace=True)

train_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                     's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                     's15', 's16', 's17', 's18', 's19', 's20', 's21']
train_df = train_df.sort_values(['id','cycle'])



# read ground truth data - It contains the information of true remaining cycles for each engine in the testing data.
truth_df = pd.read_csv(rul_path, sep=" ", header=None)
truth_df.drop(truth_df.columns[[1]], axis=1, inplace=True)

# Data Labeling - generate column RUL
rul = pd.DataFrame(train_df.groupby('id')['cycle'].max()).reset_index()
rul.columns = ['id', 'max']
train_df = train_df.merge(rul, on=['id'], how='left')
train_df['RUL'] = train_df['max'] - train_df['cycle']
train_df.drop('max', axis=1, inplace=True)

# MinMax normalization
train_df['cycle_norm'] = train_df['cycle']
cols_normalize = train_df.columns.difference(['id','cycle','RUL'])
min_max_scaler = preprocessing.MinMaxScaler()
max_values = train_df.max()
norm_train_df = pd.DataFrame(min_max_scaler.fit_transform(train_df[cols_normalize]), 
                             columns=cols_normalize, 
                             index=train_df.index)

join_df = train_df[train_df.columns.difference(cols_normalize)].join(norm_train_df)
train_df = join_df.reindex(columns = train_df.columns)
 
def gen_sequence(id_df, seq_length, seq_cols):
   
    data_array = id_df[seq_cols].values
    num_elements = data_array.shape[0]
    
    # Iterate over two lists in parallel.
    # For example id1 have 192 rows so zip iterate over two following list of numbers (0,112),(50,192)
    # 0 50 from row 0 to row 50
    # 1 51 from row 1 to row 51
    # 2 52 from row 2 to row 52
    # ...
    # 111 191 from row 111 to 191
    for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
        yield data_array[start:stop, :]

# pick the feature columns 
sensor_cols = ['s' + str(i) for i in range(1,22)]

#Dismiss following sensors and settings
sensor_cols.remove('s1')
sensor_cols.remove('s5')
sensor_cols.remove('s10')
sensor_cols.remove('s16')
sensor_cols.remove('s18')
sensor_cols.remove('s19')
sequence_cols = ['setting1', 'setting2', 'setting3', 'cycle_norm']
sequence_cols.extend(sensor_cols)
sequence_cols.extend(['RUL'])


val=list(gen_sequence(train_df[train_df['id']==1], sequence_length, sequence_cols))
#print(len(val))

#Filter Coloumns for max Values
max_values = max_values[sensor_cols]
max_values = np.array(max_values)

print(train_df.shape[0])
# transform each id of the train dataset in a sequence
seq_gen = (list(gen_sequence(train_df[train_df['id']==id], sequence_length, sequence_cols)) 
           for id in train_df['id'].unique())

# generate sequences and convert to np array
seq_array = np.concatenate(list(seq_gen)).astype(np.float32)
print(seq_array.shape[0])
max_value_rul = np.max(seq_array[:,:,19])
#label_array *= 1/max_value_rul #normalize RUL Labels
seq_array[:,:,19] *= 1/max_value_rul #normalize RUL Labels
label_array = seq_array[:,:,19]
label_array = np.reshape(label_array, (seq_array.shape[0],seq_array.shape[1],1))
seq_array = seq_array[:,:,0:-1] #drop rul column
print(seq_array.shape[0])



#return seq_array, label_array, max_value_rul, max_values
