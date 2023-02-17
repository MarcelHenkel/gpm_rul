# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 09:25:03 2022

@author: Marcel Henkel
"""

import pandas as pd
from sklearn import preprocessing
import numpy as np



###Preprocessing CMAPSS Data, by dropping unused columns, normalising and calculating RUL Labelvektor###
###Returning preprocessed Data as Sequences with length sl, beginning with [0,sl-1]00 measurement, then [1,sl] measurement,
### and so on. Stopping at [-1-sl, sl] measurment of an engine and continue with the next engine, RUL Label Vector and Max RUL Value###  

path='C:/Users/Marcel Henkel/Desktop/conditional GAN/Data/CMAPSS/'


def preprocessing_cmapps(dataset_n = 'FD001', seq_length = 24):
    path_train = path + 'train_'  + dataset_n + '.txt'
    path_test = path + 'test_'  + dataset_n + '.txt'
    
    #def preprocessing_cmapps(sl=30, path='C:/Users/Marcel Henkel/Desktop/conditional GAN/Data/CMAPSS/', short=False): #rul_path = 'C:/Users/Marcel Henkel/Desktop/conditional GAN/Data/CMAPSS/RUL_FD001.txt',
    sequence_length = seq_length
    
    train_df = pd.read_csv(path_train, sep=" ", header=None)
    #print(train_df)
    train_df.drop(train_df.columns[[26, 27]], axis=1, inplace=True)
    
    train_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                         's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                         's15', 's16', 's17', 's18', 's19', 's20', 's21']
    train_df = train_df.sort_values(['id','cycle'])
    
    
    
    # read ground truth data - It contains the information of true remaining cycles for each engine in the testing data.
    #truth_df = pd.read_csv(rul_path, sep=" ", header=None)
    #truth_df.drop(truth_df.columns[[1]], axis=1, inplace=True)
    
    # Data Labeling - generate column RUL
    rul_train = pd.DataFrame(train_df.groupby('id')['cycle'].max()).reset_index()
    rul_train.columns = ['id', 'max']
    train_df = train_df.merge(rul_train, on=['id'], how='left')
    train_df['RUL'] = train_df['max'] - train_df['cycle']
    train_df.drop('max', axis=1, inplace=True)
    
    
    train_df['cycle_norm'] = train_df['cycle']
    #cols_normalize_train = train_df.columns.difference(['id','cycle','RUL'])
    
    
    ###############################################################################
    
    test_df = pd.read_csv(path_test, sep=" ", header=None)
    #print(train_df)
    test_df.drop(test_df.columns[[26, 27]], axis=1, inplace=True)
    
    test_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                         's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                         's15', 's16', 's17', 's18', 's19', 's20', 's21']
    test_df = test_df.sort_values(['id','cycle'])
    
    
    
    # read ground truth data - It contains the information of true remaining cycles for each engine in the testing data.
    #truth_df = pd.read_csv(rul_path, sep=" ", header=None)
    #truth_df.drop(truth_df.columns[[1]], axis=1, inplace=True)
    
    # Data Labeling - generate column RUL
    rul_test = pd.DataFrame(test_df.groupby('id')['cycle'].max()).reset_index()
    rul_test.columns = ['id', 'max']
    
    #add max id from train_df to get continous id for train_df and test_df 
    rul_test['id'] = rul_test['id'] + rul_train['id'].max()
    test_df['id'] = test_df['id'] + rul_train['id'].max()
    
    #this is the id which splits train_df and test_df later again
    split_id = rul_test['id'].min()
    
    test_df = test_df.merge(rul_test, on=['id'], how='left')
    test_df['RUL'] = test_df['max'] - test_df['cycle']
    test_df.drop('max', axis=1, inplace=True)
    
    
    test_df['cycle_norm'] = test_df['cycle']
    #cols_normalize_test = test_df.columns.difference(['id','cycle','RUL'])
    
    
    ###############################################################################
    
    #concatenate train and test datasets before MinMaxScalar
    
    full_df = pd.concat([train_df,test_df],axis=0)
    full_rul = pd.concat([rul_train,rul_test], axis=0)
    
    del train_df,test_df,rul_train,rul_test
    
    
    #calculate RUL for every column
    
    full_df = full_df.merge(full_rul, on=['id'], how='left')
    
    
    full_df['RUL'] = full_df['max'] - full_df['cycle']
    full_df.drop('max', axis=1, inplace=True)
    
    # MinMax normalization
    full_df['cycle_norm'] = full_df['cycle']
    cols_normalize = full_df.columns.difference(['id','cycle','RUL'])
    
    max_values = full_df.max()
    
    
    ###############################################################################
    
    
    # MinMax normalization
    
    
    min_max_scaler = preprocessing.MinMaxScaler()
    
    norm_df = pd.DataFrame((full_df[cols_normalize]), 
                                 columns=cols_normalize, 
                                 index=full_df.index)
    
    norm_df = pd.DataFrame(min_max_scaler.fit_transform(full_df[cols_normalize]), 
                                 columns=cols_normalize, 
                                 index=full_df.index)
    
    join_df = full_df[full_df.columns.difference(cols_normalize)].join(norm_df)
    
    full_df = join_df.reindex(columns = full_df.columns)
    
    del join_df,norm_df
    
    ###############################################################################
    
    #generate sequences in length of sl
    
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
    '''
    sensor_cols.remove('s6')
    
    sensor_cols.remove('s8')
    sensor_cols.remove('s9')
    sensor_cols.remove('s13')
    sensor_cols.remove('s14')
    '''
    sensor_cols.remove('s10')
    sensor_cols.remove('s16')
    sensor_cols.remove('s18')
    sensor_cols.remove('s19')
    sequence_cols = ['setting1', 'setting2', 'setting3', 'cycle_norm']
    sequence_cols.extend(sensor_cols)
    sequence_cols.extend(['RUL'])
    
    

    #Filter Coloumns for max Values
    max_values = max_values[sensor_cols]
    max_values = np.array(max_values)

    
    ###############################################################################
    #full_df ist zu groß, splitte jetzt schon auf train und val nach split id
    
    max_id = full_rul['id'].max()
    
    train_data_1 = full_df.loc[(full_df['id']<=(int(split_id)))]
    
    
    val_data = full_df.loc[(full_df['id']<=(int((max_id-split_id)/2+split_id)))]
    val_data = val_data.loc[(val_data['id']>(int(split_id)))]
    test_data = full_df.loc[(full_df['id']<=(int(max_id)))]
    test_data = test_data.loc[(test_data['id']>(int((max_id-split_id)/2+split_id)))]
    
    
    del full_df
    
    #generate time series sequences for all 4 subsets
    seq_gen = (list(gen_sequence(train_data_1[train_data_1['id']==id], sequence_length, sequence_cols)) 
               for id in train_data_1['id'].unique())
    
    # generate sequences and convert to np array
    train_data = list(seq_gen)
    train_data = [x for x in train_data if x != []]
    train_data = np.concatenate(train_data).astype(np.float32)
    train_label = train_data[:,:,-1] #get rul col
    train_data = train_data[:,:,0:-1] #drop rul column
    
    
    
    seq_gen = (list(gen_sequence(val_data[val_data['id']==id], sequence_length, sequence_cols)) 
               for id in val_data['id'].unique())
    #in val_data1 und val_data2 sind sequenzen die weniger als 24 zeiteinheiten lang sind und daher 
    #leer sind. passe gen_sequence an, damit die ids mit weniger als seq cycle.max haben
    #übersprungen werden
    # generate sequences and convert to np array
    val_data = list(seq_gen)
    val_data = [x for x in val_data if x != []]
    val_data = np.concatenate(val_data).astype(np.float32)
    val_label = val_data[:,:,-1] #get rul col
    val_data = val_data[:,:,0:-1] #drop rul column
    
    
    seq_gen = (list(gen_sequence(test_data[test_data['id']==id], sequence_length, sequence_cols)) 
               for id in test_data['id'].unique())
    #in val_data1 und val_data2 sind sequenzen die weniger als 24 zeiteinheiten lang sind und daher 
    #leer sind. passe gen_sequence an, damit die ids mit weniger als seq cycle.max haben
    #übersprungen werden
    # generate sequences and convert to np array
    test_data = list(seq_gen)
    test_data = [x for x in test_data if x != []]
    test_data = np.concatenate(test_data).astype(np.float32)
    test_label = test_data[:,:,-1] #get rul col
    test_data = test_data[:,:,0:-1] #drop rul column
    
    
    
    ###############################################################################
    
    
    '''
    if(short):
        slicing = np.r_[0:100,1000:1100,2000:2100,3000:3100,4000:4100,5000:5100,6000:6100,7000:7100,8000:8100,9000:9100,10000:10100,11000:11100,12000:12100,13000:13100,14000:14100,15000:15100]#,16000:16100,17000:17100,18000:18100,19000:19100]
        seq_array = seq_array[slicing,:,0:19]
        label_array = label_array[slicing,:,:]
    '''
    
    return train_data, train_label, val_data,val_label, test_data, test_label
    
    
