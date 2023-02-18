# -*- coding: utf-8 -*-
"""
Created on Tue May 24 13:38:10 2022

@author: Marcel Henkel
"""
#Das NN braucht eine Sequenz von sukzessiven Zeitenreihen und als Label nur die RUL-Information der letzten Zeitreihe nomiert
#Einen Datensatz für Training und einen für die Erstellung des Histogrammes und den Score, MSE
import tensorflow as tf
import keras
import keras.backend as K
from keras.layers.core import Activation
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, LSTM, InputLayer, GRU, SimpleRNN, ConvLSTM1D, Bidirectional
from IPython.display import SVG, clear_output
#from keras.utils import plot_model
from keras.utils.vis_utils import plot_model #updated
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import tensorflow.keras.optimizers as optimizer
from sklearn import preprocessing
import data_preprocessing_cmapps as cdp
import data_preprocessing_battery as badp
import VAE
from keras import layers
from tensorflow.keras import activations
from sklearn.metrics import mean_squared_error
import data_preprocessing_bearing as bdp
#K.tensorflow_backend._get_available_gpus() #habe keine GPU / noch nicht

#Hyperparameter 
use_VAE = False
np.random.seed(1234)  
PYTHONHASHSEED = 0
sequence_length =seq_length= 24
epochen = 1000
lr_rul = 0.001
batch_rul = 128
early_stop_patience = 100

# define path to save model
model_path = '4regression_model.h5'


dataset_n = 'FD001'
flat_rul = True

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

#train_data, train_label, val_data,val_label, test_data, test_label, max_value_rul = battery(sequence_length)
train_data, train_label, val_data,val_label, test_data, test_label, max_value_rul = bearing(sequence_length, False, 120, flat = True)
#train_data, train_label, val_data,val_label, test_data, test_label, max_value_rul = cmapss(sequence_length,dataset_n)
#shortening training data
#[0:100,1000:1100,2000:2100,3000:3100] 400
#[0:100,1000:1100]
#train_data = train_data[43:167,:,:]
#train_label = train_label[43:167,:]

if(False):
    #slicing = np.r_[100:150]#,2000:2100,3000:3100]#,4000:4100,5000:5100,6000:6100,7000:7100,8000:8100,9000:9100,10000:10100,11000:11100,12000:12100,13000:13100,14000:14100,15000:15100]#,16000:16100,17000:17100,18000:18100,19000:19100]
    slicing = np.r_[0:200]#,2000:2100,3000:3100]#,4000:4100,5000:5100,6000:6100,7000:7100,8000:8100,9000:9100,10000:10100,11000:11100,12000:12100,13000:13100,14000:14100,15000:15100]#,16000:16100,17000:17100,18000:18100,19000:19100]
    #slicing = np.r_[0:100,1000:1100]#,2000:2100,3000:3100]#,4000:4100,5000:5100,6000:6100,7000:7100,8000:8100,9000:9100,10000:10100,11000:11100,12000:12100,13000:13100,14000:14100,15000:15100]#,16000:16100,17000:17100,18000:18100,19000:19100]
    #slicing = np.r_[0:100,1000:1100,2000:2100,3000:3100]#,4000:4100,5000:5100,6000:6100,7000:7100,8000:8100,9000:9100,10000:10100,11000:11100,12000:12100,13000:13100,14000:14100,15000:15100]#,16000:16100,17000:17100,18000:18100,19000:19100]
    #slicing = np.r_[0:100,1000:1100,2000:2100,3000:3100,4000:4100,5000:5100,6000:6100,7000:7100,8000:8100,9000:9100,10000:10100,11000:11100,12000:12100,13000:13100,14000:14100,15000:15100]#,16000:16100,17000:17100,18000:18100,19000:19100]
    train_data = train_data[slicing,:,:]
    train_label = train_label[slicing,:]

# read additionl synthetic data 
if(True): #if True load numpy dataset with RUL information in the last column, else use data_preprocessing function to load data 
    outfile = 'synthetic datasets/' + '24Bearing_fake_data_100_cr_gan.npy' #add Model Info to title
    train_data = np.load(outfile)
    outfile = 'synthetic datasets/' + '24Bearing_fake_data_label_100_cr_gan.npy' #add Model Info to title
    train_label = np.load(outfile)
    #s_label = s_label[:,seq_length-1,:]
    #concate with original trainingdata
    #train_data = np.concatenate([train_data,s_data],axis=0)
    
    
    #val_label = val_label[:,sequence_length-1,:]
    #test_label = test_label[:,sequence_length-1,:]

def r2_keras(y_true, y_pred):
   
    res =  K.sum(K.square( y_true - y_pred ))
    tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - res/(tot + K.epsilon()) )

from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

opt = optimizer.Adam(learning_rate=0.001,beta_2=0.999)
sequence_length = train_data.shape[1] #Länge einer Messsequenz

# Next, we build a deep network. 
# The first layer ==> LSTM layer with 100 units
# Second Layer ==> Convolution Layer
# Third Layer ==>  LSTM layer with 50 units. 
# Dropout is applied after each LSTM layer to control overfitting. 
# Final layer is a Dense output layer with single unit and Relu activation.
nb_features = train_data.shape[2]
nb_out = train_label.shape[1]

model = Sequential()
model.add(InputLayer(input_shape=(train_data.shape[1], train_data.shape[2])))

model.add((GRU(units=80,return_sequences=True,kernel_regularizer=tf.keras.regularizers.L2(l2=0.00001))))
model.add(Dropout(0.2))

model.add((GRU(units=40,return_sequences=True)))
model.add(Dropout(0.2))

model.add((GRU(units=20,return_sequences=False)))
model.add(Dropout(0.1))



'''
#model.add(LSTM(units=15,return_sequences=True))
#model.add(Dropout(0.2))
'''
#model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
#model.add(MaxPooling1D(pool_size=2))

#model.add(LSTM(units=10,return_sequences=False))
#model.add(Dropout(0.2))
model.add(Dense(units = 20))
#model.add(Dense(units = 10))
#model.add(Dense(units = 5))
model.add(Dense(units=1))
model.add(layers.Activation(activations.relu))
model.compile(loss='mse', optimizer='adam',metrics=['mse',r2_keras])

#optimizer rmsprop oder adam

print(model.summary())

# plot
class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []        
        self.fig = plt.figure()        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        
        clear_output(wait=True)
        plt.plot(self.x, np.sqrt(self.losses), label="loss")
        plt.plot(self.x, np.sqrt(self.val_losses), label="val_loss")
        plt.ylabel('loss - RMSE')
        plt.xlabel('epoch')
        plt.legend(['train','test'], loc='upper left')
        plt.title('model loss = ' + str(min(np.sqrt(self.val_losses))))
        plt.show();
        
plot_losses = PlotLosses()
# Define learning rate
K.set_value(model.optimizer.learning_rate, lr_rul)


my_callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss",verbose=1, restore_best_weights=True, patience=early_stop_patience),
                #keras.callbacks.ModelCheckpoint(model_path,monitor='val_loss', save_best_only=True, mode='min', verbose=0)
                ]

#validation_split=0.05,
# fit the network
history = model.fit(train_data, train_label, epochs=10, batch_size=batch_rul, validation_data=(val_data,val_label), verbose=2,)

history = model.fit(train_data, train_label, epochs=epochen, batch_size=batch_rul, validation_data=(val_data,val_label), verbose=2,
          callbacks = my_callbacks)
#deleted keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='min'),
# list all data in history
print(history.history.keys())


# summarize for R^2
fig_acc = plt.figure(figsize=(10, 10))
plt.plot(history.history['r2_keras'])
plt.plot(history.history['val_r2_keras'])
plt.title('model r^2')
plt.ylabel('R^2')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
fig_acc.savefig("model_r2.png")

# summarize for MsE
fig_acc = plt.figure(figsize=(6, 6))
#plt.plot(history.history['mean_absolute_error'])
#plt.plot(history.history['val_mean_absolute_error'])
plt.plot(history.history['mse'])
plt.plot(history.history['val_mse'])

plt.title('RNN RUL model losses')
plt.ylabel('MSE')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
fig_acc.savefig("model_mse.png")

# summarize for Loss
fig_acc = plt.figure(figsize=(6, 6))
#plt.plot(save_train_loss)
#plt.plot(save_val_loss)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('RNN RUL model loss')
plt.ylabel('loss [MSE]')
plt.xlabel('epoch')
plt.legend(['train loss', 'validation loss'], loc='center right')
plt.show()
fig_acc.savefig("model_regression_loss.png")

# training metrics
scores = model.evaluate(train_data, train_label, verbose=1, batch_size=200)
print('\nMSE: {}'.format(scores[1]))
print('\nR^2: {}'.format(scores[2]))

# training metrics
scores = model.evaluate(val_data, val_label, verbose=1, batch_size=200)
print('\nMSE: {}'.format(scores[1]))
print('\nR^2: {}'.format(scores[2]))


# transform each id of the train dataset in a sequence
def mae(y_pred,y_true):
    return np.sum(np.abs((y_pred-y_true)))/y_pred.shape[0]

def score_cal(y_hat, Y_test):
    d   = y_hat - Y_test
    tmp = np.zeros(d.shape[0])
    for i in range(d.shape[0]):
        if d[i,0] >= 0:
           tmp[i] = np.exp( d[i,0]/10) - 1
        else:
           tmp[i] = np.exp(-d[i,0]/13) - 1
    return tmp 


if True:
    #test metrics
    #scores_test = model.evaluate(seq_array_test_last, label_array_test_last, verbose='auto')
    #print('\nMSE: {}'.format(scores_test[1]))
    #print('\nR^2: {}'.format(scores_test[2]))
    y_pred_val = model.predict(val_data)
    y_pred_test = model.predict(test_data)

    
    val_mse = mean_squared_error(y_pred_val, val_label)
    test_mse = mean_squared_error(y_pred_test, test_label)
    train_mse = history.history['loss'][-1]
    val_mae = mae(y_pred_val,val_label)
    test_mae = mae(y_pred_test, test_label)
    print('val_mse: ' + str(val_mse))
    print('val_mae: ' + str(val_mae)) 
          
    print('test_mse: ' + str(test_mse))
    print('test_mae: ' + str(test_mae))        

    '''
    if(False):
        rul_offset_test = np.repeat(rul_offset[np.newaxis,:],y_pred_test.shape[0],axis=0)
        rul_offset_test = rul_offset_test[:,:,np.newaxis]
        rul_l = []
        for x in range(y_pred_test.shape[0]):
            rul_l.append(np.sum([y_pred_test[x,:,:]-rul_offset_test[x,:,:]])/seq_length)
        y_pred_test = np.array(rul_l)
        y_pred_test = y_pred_test[:,np.newaxis]
    else: 
        y_pred_test = y_pred_test[:,seq_length-1,:]
    '''  
    


    battery_val_row_1 = 166
    battery_val_row_2 = 205
    
    true_val_label_1 = val_label[0:battery_val_row_1,:]
    pred_val_label_1 = y_pred_test[0:battery_val_row_1,:]
    
    true_val_label_2 = val_label[battery_val_row_1:battery_val_row_2,:]
    pred_val_label_2 = y_pred_test[battery_val_row_1:battery_val_row_2,:]
    
    true_val_label_3 = val_label[battery_val_row_2:-1,:]
    pred_val_label_3= y_pred_test[battery_val_row_2:-1,:]
    #in die Überschrift der plots namen des datensatzes aufführen

    
    #split and plot battery testset

    fig_verify = plt.figure(figsize=(10, 5))
    #plt.plot(pred_val_label_1*max_value_rul)#, color="orange")
    #plt.plot(true_val_label_1*max_value_rul)#, color="blue")
    #plt.plot(pred_val_label_2*max_value_rul)#, color="orange")
    #plt.plot(true_val_label_2*max_value_rul)#, color="blue")
    #plt.plot(pred_val_label_3*max_value_rul)#, color="orange")
    #plt.plot(true_val_label_3*max_value_rul)#, color="blue")
    plt.plot(train_label*max_value_rul, color="green")

    #plt.title('Predicted and real remaining capaicty')
    plt.title('predicted and real remaining capacity (test set)')
    plt.ylabel('remaining capacity [Ah]')
    plt.xlabel('cylce')
    #plt.legend(['pred bat1', 'real bat1','pred bat2', 'real bat2','pred bat3', 'real bat3'], loc='upper right')
    #plt.legend([ 'prediction battery B0006','real battery B0006','prediction battery B0034', 'real battery B0034','prediction battery B0056','real battery B0056'], loc='lower right')
    plt.legend([ 'battery B0006'])
    plt.show()

    '''
    #plot bearing testset
    x_values = np.linspace(0,100,num=599)
    fig_verify = plt.figure(figsize=(10, 5))
    plt.plot(x_values,y_pred_test*max_value_rul, color="blue")
    plt.plot(x_values,test_label*max_value_rul, color="green")
    plt.title('predicted and real remaining runtime (test set)')
    plt.ylabel('RUL [h]')
    plt.xlabel('runtime [h]')
    plt.legend(['predicted RUL bearing 4', 'real RUL bearing 4'], loc='upper right')
    plt.show()
    
    x_values = np.linspace(0,100,num=599)
    fig_verify = plt.figure(figsize=(10, 5))
    plt.plot(x_values,y_pred_val*max_value_rul, color="blue")
    plt.plot(x_values,val_label*max_value_rul, color="green")
    plt.title('predicted and real remaining runtime (validation set)')
    plt.ylabel('RUL [h]')
    plt.xlabel('runtime [h]')
    plt.legend(['predicted RUL bearing 3', 'real RUL bearing 3'], loc='upper right')
    plt.show()

    '''
    '''
    #cmapps
    fig_verify = plt.figure(figsize=(10, 5))
    plt.plot(y_pred_val[2700:4800,:]*max_value_rul, color="blue")
    plt.plot(val_label[2700:4800,:]*max_value_rul, color="green")
    plt.title('predicted and real RUL cmapps (validation set FD004)')
    plt.ylabel('RUL [cycle]')
    plt.xlabel('cycles')
    plt.legend(['predicted RUL', 'actual data RUL'], loc='upper left')
    plt.show()
    '''
    '''
    d = y_pred_val*max_value_rul - val_label*max_value_rul
    plt.hist(d, bins='auto')  
    plt.title('Error distribution - test set FD001')
    plt.ylabel('f')
    plt.xlabel("Error: $RUL_{hat}$ - RUL [cycle]")
    plt.show()
    '''
