# -*- coding: utf-8 -*-
"""
Created on Mon May 16 17:01:21 2022

@author: Marcel Henkel
"""

from os import walk
import pandas as pd
import numpy as np
from numpy.fft import fft, fftfreq
import matplotlib.pyplot as plt
from scipy import log
from scipy import signal
from scipy.fft import fftshift
from sklearn import preprocessing

#parameter der finalen methode seq_length, auflösung fft
#am besten datensatz speichern, denn die Verarbeitung dauert sehr lange, als np.array speichern [train_data,train_label,val_data,val_label,test_data,test_label]


raw_data_paths = ['Data/Bearing/IMS/1st_test/', 'Data/Bearing/IMS/2nd_test/', 'Data/Bearing/IMS/4th_test/txt/']
save_path = 'IMS/'

# res Length of each segment in hz => higher number lower res, lower number higher resolution
def preprocessing_bearing(sequence_length, log, freq_gab):
    filenames_arr = []
    for r in raw_data_paths:
        filenames = next(walk(r), (None, None, []))[2]  
        filenames_arr.append(filenames)
    #failures=[[4,6],[0],[2]] #indices of bearings with failure
    b_ch_non_failures= [[1,2,3,5,7],[1,2,3],[0,1,3]] #indices of bearings without failure
    num_csv = len(filenames) #number of csv files
    #calculate exact run-time or get it from analysis for each test-run
    sq_len = sequence_length
    
    '''
    Info Dataset:
        
        1st Test: Faliures: Bearing 3: inner race, Bearing 4: rolling element, len filenames = 2135 => Channel 4,6 oder Channel 5,7
        2nd Test: Faliures: Bearing 1: outer race, len filenames => Chanel 0
        3rd Test: Bearing 3: outer race, len filenames => Chanel 2
        
        every Column in the org file is representing a channel
    '''
    
    #Gebe RUL in der Einheit Stunden an.
    
    
    cycle_col_arr = []
    rul_col_arr = []
    for f in filenames_arr:
        cycle_col = np.round(np.arange(len(f))*10/60,1)
        rul_col= np.flip(cycle_col)
        cycle_col_arr.append(cycle_col)
        rul_col_arr.append(rul_col)
        
    
    #rul_col_arr => list of 3 np array for the 3 runs
    #only failing bearings
    fail = 1 #1 if Failure after rul, or 1 if no Failure after rul
    b_pos_list = [[3,4],[1],[3]] #bearing position
    b_ch_list = [[4,6],[0],[2]] #coresponding bearing channel to bearing positionan
    
    #dataf = np.array(np.arange(127)) #storage for all fft and additional data for a Bearing in one testrun
    
    
    dataf = []
    
    id_b = 0
    for i in range(3): #for every run 1,2,3
        filenames = next(walk(raw_data_paths[i]), (None, None, []))[2]
        rul = rul_col_arr[i]
        cycle = cycle_col_arr[i]
        run_i = 0 #tracking index for every run (1st,2nd and 3rd)
        
        for x in range(len(b_ch_list[i])): #for each (failed) bearing per run (b_ch_list), list can be extended to non failing bearings 
            print('i: ' + str(i))
            print('x: ' + str(x))
            id_b = id_b +1
            b_ch = b_ch_list[i][x]
            b_pos = b_pos_list[i][x]
            for f in filenames: #all filenames of one test (1st, 2nd or 3rd), i = 0 => 1st, i = 1 => 2nd, i =2 => 3rd
                all_m = pd.DataFrame() #time series data will be stored there and reset every loop
                
                #index_b = b_ch
                #index_b = failures[i] #Index or column in Dataset: 0 => Bearing 1 x Axis, 1 => Bearing 1 y Axis, 2 => Bearing 2 x Axis 
                df = pd.read_csv(raw_data_paths[i]+f, sep='\t', header=None) 
                #for b in index_b: #falls in einem Lauf mehrere bearings gefailt haben (1st run)
                #die for schleife wurde durch forschleife in f x in range(len(b_ch_list[i])): ersetzt
                    
                b1 = df.iloc[:,[b_ch]] 
                all_m = pd.concat([all_m,b1]) #create one var with all values of all measurements 
                all_m = all_m.values.flatten() #make 1D dataframe of timeseries Acceraltion Values from Sensor
        
                #rng = np.random.default_rng() #unused??
        
                #Fourier Transformation https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.spectrogram.html
                fs = 20e3 #Sampling frequency of the time series = samplingrate (20kHz)
    
                res= freq_gab
                f, t, Sxx = signal.spectrogram(all_m, fs, nperseg=res) #fft 
        
                #slicing desired frequency-window
                f_min=0 #min frequency
                f_max=4000 #max frequency
                freq_slice = np.where((f >= f_min) & (f <= f_max))
        
                # keep only frequencies of interest
                f   = f[freq_slice]
                Sxx = Sxx[freq_slice,:][0]
                
                #Sxx_sum = pd.DataFrame(np.sum(Sxx, axis=1)) #nehme Zeit heraus und Summiere die Amplituden der Frequenzen über die Zeit zu einem Wert 
                Sxx_sum = np.sum(Sxx, axis=1)
                #convert to a row of data
                Sxx_sum = np.reshape(Sxx_sum,[1,Sxx_sum.shape[0]])
                
                
                
                
                #addtional information for a row / measurment
                #RUL [h], runtime [h], fail [0,1], bearing position, test number, id
                
                a_info = np.array([rul[run_i],cycle[run_i], fail, b_pos, str(i+1), id_b])
                a_info = np.reshape(a_info, [1,a_info.shape[0]])
                
                row = np.concatenate([a_info,Sxx_sum], axis=1)
                datatf = np.array(np.arange(row.shape[1])) #storage for all fft and additional data for a Bearing in one testrun
                datatf = np.reshape(datatf,[1,datatf.shape[0]])
    
                #concat every row to dataf
                dataf.append(np.concatenate([datatf,row],axis=0))
                run_i=run_i+1
            run_i = 0
    #convert list into np array and delete every frist row, because it is unwanted
    dataf = np.array(dataf,dtype=np.float64)
    dataf = dataf[:,1,:]
    #
    min_max_scaler = preprocessing.MinMaxScaler()
    
    label = dataf[:,0:6]
    data = dataf[:,6:dataf.shape[1]]
    
    #drop first row
    label = label[1:label.shape[0],:]
    data = data[1:data.shape[0],:]
    
    #use log function on each value
    if(log):
        data = np.array(data, dtype=np.float64)
        data = np.log(data)
    
    #min max scalar
    data = min_max_scaler.fit_transform(data)
    
    #concate rul label and data
    data = np.concatenate([label,data],axis=1)
    
    #set data type to float64
    data = np.array(data, dtype=np.float64)
    
    
    del dataf, b1, cycle, cycle_col_arr, rul, rul_col, rul_col_arr, label
    #generate sequences 
    
    def gen_sequence(data_from_id, seq_length):
        
        length_of_data = data_from_id.shape[0]
        
        
        for i in range(length_of_data-seq_length):
            temp = data_from_id[i:i+seq_length]
            temp = np.reshape(temp, [1,temp.shape[0],temp.shape[1]])
            if i == 0:
                seq = temp
            else:
                seq = np.concatenate([seq,temp],axis=0)
        # Iterate over two lists in parallel.
        # For example id1 have 192 rows so zip iterate over two following list of numbers (0,112),(50,192)
        # 0 50 from row 0 to row 50
        # 1 51 from row 1 to row 51
        # 2 52 from row 2 to row 52
        # ...
        # 111 191 from row 111 to 191
        return seq
    
    seq_data = []
    
    max_id = np.max(data[:,5])
    
    for i in range(int(max_id)):
        row_same_id = np.argwhere(data[:,5]==i+1)
        data_same_id = data[int(row_same_id[0]):int(row_same_id[-1]),:]
        seq_data.append(gen_sequence(data_same_id, sq_len))
        
    seq_data_np = seq_data[0]
    
    for i in range(1,len(seq_data)):
        seq_data_np = np.concatenate([seq_data_np,seq_data[i]],axis=0)
    
    
    del data, seq_data
    
    #split in training-, validation- und test-data 
    
    #train_data =>id = 1,2 val_data id = 3 and test_data id  = 4
    
    row_same_id = np.argwhere(seq_data_np[:,0,5]==1)
    train_data = seq_data_np[int(row_same_id[0]):int(row_same_id[-1]),:,:]
    
    row_same_id = np.argwhere(seq_data_np[:,0,5]==2)
    train_data = np.concatenate([train_data,seq_data_np[int(row_same_id[0]):int(row_same_id[-1]),:,:]],axis=0)
    
    row_same_id = np.argwhere(seq_data_np[:,0,5]==3)
    val_data = seq_data_np[int(row_same_id[0]):int(row_same_id[-1]),:,:]
    
    row_same_id = np.argwhere(seq_data_np[:,0,5]==4)
    test_data = seq_data_np[int(row_same_id[0]):int(row_same_id[-1]),:,:]
    
    del seq_data_np
    
    train_label = train_data[:,:,0:1]#np.reshape(train_data[:,:,0:1],[train_data[:,:,0].shape[0],train_data[:,:,0].shape[1],1])
    val_label = val_data[:,:,0:1]#np.reshape(val_data[:,:,0:1],[val_data[:,:,0].shape[0],val_data[:,:,0].shape[1],1])
    test_label = test_data[:,:,0:1]#np.reshape(test_data[:,:,0:1],[test_data[:,:,0].shape[0],test_data[:,:,0].shape[1],1])
    
    #drop rul column
    train_data = train_data[:,:,1:train_data.shape[2]]
    val_data = val_data[:,:,1:val_data.shape[2]]
    test_data = test_data[:,:,1:test_data.shape[2]]
    
    max_rul = np.max(np.concatenate([train_label[:,0,0],test_label[:,0,0],val_label[:,0,0]],axis=0))
    
    return train_data, train_label, val_data,val_label, test_data, test_label, max_rul

