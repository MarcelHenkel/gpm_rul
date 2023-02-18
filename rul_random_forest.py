# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 22:00:27 2022

@author: Marcel Henkel
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn import preprocessing
from sklearn.metrics import r2_score
import numpy as np
import data_preprocessing_cmapps as cdp
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import data_preprocessing_cmapps as cdp
import data_preprocessing_bearing as bdp
import data_preprocessing_battery as badp
from sklearn.ensemble import RandomForestRegressor

seq_length = sequence_length = 24

dataset_n = 'FD001'
flat_rul = True
max_res = 10

def battery(seq_length):
    return badp.preprocessing_battery(seq_length)

def cmapss(seq_length, dataset_n):

    
    train_data, train_label, val_data,val_label, test_data, test_label = cdp.preprocessing_cmapps(dataset_n=dataset_n, seq_length = seq_length)
    
    '''
    train_label = train_label[:,sequence_length-1]
    val_label = val_label[:,sequence_length-1]
    test_label = test_label[:,sequence_length-1]
    '''
    max_value_rul = np.amax(np.array([np.amax(train_label),np.amax(val_label),np.amax(test_label)]))    
    
    
    #delete cylce column
    train_data = np.delete(train_data, 3, axis=2) 
    val_data = np.delete(val_data, 3, axis=2)
    test_data = np.delete(test_data, 3, axis=2)
       
    #delete setting column
    
    train_data = np.delete(train_data, 6, axis=2) 
    val_data = np.delete(val_data, 6, axis=2)
    test_data = np.delete(test_data, 6, axis=2)
    
    train_data = np.delete(train_data, 2, axis=2) 
    val_data = np.delete(val_data, 2, axis=2)
    test_data = np.delete(test_data, 2, axis=2)
    
    #undo rul normalisation
    #train_label = train_label*max_value_rul
    #val_label = val_label*max_value_rul
    #test_label = test_label*max_value_rul
    
    if flat_rul:
        #Flatline RUL above 125 cycles to 125 cycles
        train_label[train_label>125] = 125
        val_label[val_label>125] = 125
        test_label[test_label>125] = 125
    

    
    train_label = train_label/max_value_rul
    val_label = val_label/max_value_rul
    test_label = test_label/max_value_rul
    
    m_rul = np.max(np.concatenate([train_label,val_label,test_label],axis=0))

    train_label = np.expand_dims(train_label[:,sequence_length-1],axis=1)
    val_label = np.expand_dims(val_label[:,sequence_length-1],axis=1)
    test_label = np.expand_dims(test_label[:,sequence_length-1],axis=1)
    
    train_label = (train_label/m_rul)
    val_label = (val_label/m_rul)
    test_label = (test_label/m_rul)
    
    return train_data, train_label, val_data,val_label, test_data, test_label, max_value_rul
#bearing dataset
def bearing(seq_length, log, freq_gab, flat = True):
    new = True
    if new:
        train_data, train_label, val_data,val_label, test_data, test_label, max_value_rul = bdp.preprocessing_bearing(seq_length, log, freq_gab)
        '''
        train_label = train_label[:,seq_length-1,:]
        val_label = val_label[:,seq_length-1,:]
        test_label = test_label[:,seq_length-1,:]
        '''
        
        if flat:
            #Flatline RUL above 300 hours to 300 hours
            train_label[train_label>75] = 75
            val_label[val_label>75] = 75
            test_label[test_label>75] = 75
           
        #only use last 100 measurements, when rul is limited to 75
        train_label = np.concatenate([train_label[int(train_label.shape[0]/2)-600:int(train_label.shape[0]/2),:],train_label[int(train_label.shape[0])-600:int(train_label.shape[0])-1,:]], axis=0)
        val_label = val_label[-600:-1,:]
        test_label = test_label[-600:-1,:]
        
        train_data = np.concatenate([train_data[int(train_data.shape[0]/2)-600:int(train_data.shape[0]/2),:],train_data[int(train_data.shape[0])-600:int(train_data.shape[0])-1,:]], axis=0)
        val_data = val_data[-600:-1,:,:]
        test_data = test_data[-600:-1,:,:]
        
        
        
        #min max scalar label vectors
        train_label = train_label/75
        val_label = val_label/75
        test_label = test_label/75
        
        '''
        #drop cycle col, bearing number, run number, etc
        train_data = train_data[:,:,5:train_data.shape[2]]
        val_data = val_data[:,:,5:val_data.shape[2]]
        test_data = test_data[:,:,5:test_data.shape[2]]
        '''
    else:
        
        outfile = 'Data/' + 'bearing_train_data_24_32.npy' 
        train_data = np.load(outfile)
        
        outfile = 'Data/' + 'bearing_train_label_24_32.npy' 
        train_label = np.load(outfile)
        
        outfile = 'Data/' + 'bearing_val_data_24_32.npy' 
        val_data = np.load(outfile)
        
        outfile = 'Data/' + 'bearing_val_label_24_32.npy' 
        val_label = np.load(outfile)
        
        outfile = 'Data/' + 'bearing_test_data_24_32.npy' 
        test_data = np.load(outfile)
        
        outfile = 'Data/' + 'bearing_test_label_24_32.npy' 
        test_label = np.load(outfile)
        
        
        
        
        
    #drop position col, setting, rul
        
    train_data = train_data[:,:,5:train_data.shape[2]]
    val_data = val_data[:,:,5:val_data.shape[2]]
    test_data = test_data[:,:,5:test_data.shape[2]]
    
    max_value_rul = 75
    
    train_label =train_label[:,seq_length-1,:]
    val_label = val_label[:,seq_length-1,:]
    test_label =test_label[:,seq_length-1,:]
    
    return train_data, train_label, val_data,val_label, test_data, test_label, max_value_rul

#train_data, train_label, val_data,val_label, test_data, test_label, max_value_rul = bearing(sequence_length, False, 120, flat = True)
#train_data, train_label, val_data,val_label, test_data, test_label, max_value_rul = battery(sequence_length)
train_data, train_label, val_data,val_label, test_data, test_label, max_value_rul = cmapss(sequence_length,dataset_n)
    
if(False):
    #slicing = np.r_[100:150]#,2000:2100,3000:3100]#,4000:4100,5000:5100,6000:6100,7000:7100,8000:8100,9000:9100,10000:10100,11000:11100,12000:12100,13000:13100,14000:14100,15000:15100]#,16000:16100,17000:17100,18000:18100,19000:19100]
    slicing = np.r_[0:400]#,2000:2100,3000:3100]#,4000:4100,5000:5100,6000:6100,7000:7100,8000:8100,9000:9100,10000:10100,11000:11100,12000:12100,13000:13100,14000:14100,15000:15100]#,16000:16100,17000:17100,18000:18100,19000:19100]
    #slicing = np.r_[0:100,1000:1100]#,2000:2100,3000:3100]#,4000:4100,5000:5100,6000:6100,7000:7100,8000:8100,9000:9100,10000:10100,11000:11100,12000:12100,13000:13100,14000:14100,15000:15100]#,16000:16100,17000:17100,18000:18100,19000:19100]
    #slicing = np.r_[0:100,1000:1100,2000:2100,3000:3100]#,4000:4100,5000:5100,6000:6100,7000:7100,8000:8100,9000:9100,10000:10100,11000:11100,12000:12100,13000:13100,14000:14100,15000:15100]#,16000:16100,17000:17100,18000:18100,19000:19100]
    #slicing = np.r_[0:100,1000:1100,2000:2100,3000:3100,4000:4100,5000:5100,6000:6100,7000:7100,8000:8100,9000:9100,10000:10100,11000:11100,12000:12100,13000:13100,14000:14100,15000:15100]#,16000:16100,17000:17100,18000:18100,19000:19100]
    train_data = train_data[slicing,:,:]
    train_label = train_label[slicing,:]

# read additionl synthetic data 
if(True): #if True load numpy dataset with RUL information in the last column, else use data_preprocessing function to load data 
    outfile = 'synthetic datasets/' + '24cmapss_row50_cr_gan_data.npy' #add Model Info to title
    s_data = np.load(outfile)
    outfile = 'synthetic datasets/' + '24cmapss_row50_cr_gan_data_label.npy' #add Model Info to title
    s_label = np.load(outfile)
    #s_label = s_label[:,seq_length-1,:]
    #concate with original trainingdata
    #train_data = np.concatenate([train_data,s_data],axis=0)

train_l_m = np.max(train_label)
val_l_m = np.max(val_label)
test_l_m = np.max(test_label)
lab = preprocessing.LabelEncoder()

train_label = lab.fit_transform(train_label)
val_label = lab.fit_transform(val_label)
test_label = lab.fit_transform(test_label)

train_label = np.round((train_label/np.max(train_label))*max_res,0)

train_data = np.reshape(train_data,[train_data.shape[0],train_data.shape[1]*train_data.shape[2]])
val_data = np.reshape(val_data,[val_data.shape[0],val_data.shape[1]*val_data.shape[2]])
test_data = np.reshape(test_data,[test_data.shape[0],test_data.shape[1]*test_data.shape[2]])
'''
#Define and train Model
clf = RandomForestClassifier(max_depth=30, random_state=0)
clf.fit(train_data, train_label.ravel())
'''
# fit model
clf = RandomForestRegressor(n_estimators=100)
clf.fit(train_data, train_label) #train_label.values.ravel()

def score_cal(y_hat, Y_test):
    d   = y_hat - Y_test
    tmp = np.zeros(d.shape[0])
    for i in range(d.shape[0]):
        if d[i,0] >= 0:
           tmp[i] = np.exp( d[i,0]/10) - 1
        else:
           tmp[i] = np.exp(-d[i,0]/13) - 1
    return tmp 


def mse(y_pred,y_true):
    return np.sum(np.square(y_pred-y_true))/y_pred.shape[0]

def mae(y_pred,y_true):
    return np.sum(np.abs((y_pred-y_true)))/y_pred.shape[0]


#Evaluation
if True:
    #test metrics
    #scores_test = model.evaluate(seq_array_test_last, label_array_test_last, verbose='auto')
    #print('\nMSE: {}'.format(scores_test[1]))
    #print('\nR^2: {}'.format(scores_test[2]))
    y_pred_val = clf.predict(val_data)
    y_pred_test = clf.predict(test_data)
    if False: #cmapss
        y_pred_val = (y_pred_val / np.max(y_pred_val))*125
        y_pred_test = y_pred_test / np.max(y_pred_test)*125
        
        val_label_e = val_label/np.max(val_label)*125
        test_label_e = test_label/np.max(test_label)*125
    else:
        y_pred_val = (y_pred_val / np.max(y_pred_val))*val_l_m
        y_pred_test = y_pred_test / np.max(y_pred_test)*test_l_m
        
        val_label_e = val_label/np.max(val_label)*val_l_m
        test_label_e = test_label/np.max(test_label)*test_l_m
    
    val_mse = mean_squared_error(y_pred_val, val_label_e)
    test_mse = mean_squared_error(y_pred_test, test_label_e)

    val_mae = mae(y_pred_val, val_label_e)
    test_mae = mae(y_pred_test, test_label_e)
    print('val_mse: ' + str(val_mse))
    print('val_mae: ' + str(val_mae)) 
          
    print('test_mse: ' + str(test_mse))
    print('test_mae: ' + str(test_mae))      
    #test metrics

    '''
    
    fig_verify = plt.figure(figsize=(10, 5))
    #plt.plot(y_pred[0:1000,0]*max_rul_test, color="blue")
    plt.plot(y_pred_val[0:5000], color="blue")
    plt.plot(val_label_e[0:5000], color="green")
    plt.title('Predicted and real RUL')
    plt.ylabel('RUL [cycle]')
    plt.xlabel('sequences [n]')
    plt.legend(['predicted', 'actual data'], loc='upper left')
    plt.show()
    
    
    #y = np.flip(np.arange(0,262,1))
    '''
    
    '''
    battery_val_row_1 = 166
    battery_val_row_2 = 205
    
    true_val_label_1 = test_label_e[0:battery_val_row_1]
    pred_val_label_1 = y_pred_test[0:battery_val_row_1]
    
    true_val_label_2 = test_label_e[battery_val_row_1:battery_val_row_2]
    pred_val_label_2 = y_pred_test[battery_val_row_1:battery_val_row_2]
    
    true_val_label_3 = test_label_e[battery_val_row_2:-1]
    pred_val_label_3= y_pred_test[battery_val_row_2:-1]
    #in die Überschrift der plots namen des datensatzes aufführen

    
    #split and plot battery testset

    fig_verify = plt.figure(figsize=(10, 5))
    plt.plot(pred_val_label_1*max_value_rul)#, color="orange")
    plt.plot(true_val_label_1*max_value_rul)#, color="blue")
    plt.plot(pred_val_label_2*max_value_rul)#, color="orange")
    plt.plot(true_val_label_2*max_value_rul)#, color="blue")
    plt.plot(pred_val_label_3*max_value_rul)#, color="orange")
    plt.plot(true_val_label_3*max_value_rul)#, color="blue")
    #plt.plot(train_label*max_value_rul, color="green")

    #plt.title('Predicted and real remaining capaicty')
    plt.title('predicted and real remaining capacity (test set)')
    plt.ylabel('remaining capacity [Ah]')
    plt.xlabel('cylce')
    #plt.legend(['pred bat1', 'real bat1','pred bat2', 'real bat2','pred bat3', 'real bat3'], loc='upper right')
    plt.legend([ 'prediction battery B0006','real battery B0006','prediction battery B0034', 'real battery B0034','prediction battery B0056','real battery B0056'], loc='upper right')
    #plt.legend([ 'battery B0006'])
    plt.show()

    
    #plot bearing testset
    x_values = np.linspace(0,100,num=599)
    fig_verify = plt.figure(figsize=(10, 5))
    plt.plot(x_values,y_pred_test*75, color="blue")
    plt.plot(x_values,test_label_e*75, color="green")
    plt.title('predicted and real remaining runtime (test set)')
    plt.ylabel('RUL [h]')
    plt.xlabel('runtime [h]')
    plt.legend(['predicted RUL', 'real RUL'], loc='lower left')
    plt.show()
    '''
    
    #cmapps
    fig_verify = plt.figure(figsize=(10, 5))
    plt.plot(y_pred_test[3500:4200]*120, color="blue")
    plt.plot(test_label[3500:4200], color="green")
    plt.title('predicted and real RUL cmapps (validation set FD004)')
    plt.ylabel('RUL [cycle]')
    plt.xlabel('cycles')
    plt.legend(['predicted RUL', 'real RUL'], loc='lower right')
    plt.show()
   
