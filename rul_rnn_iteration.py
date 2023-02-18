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

#from tensorflow.keras.layers.core import Activation
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM, InputLayer, GRU, SimpleRNN, ConvLSTM1D, Bidirectional
from IPython.display import SVG, clear_output
#from keras.utils import plot_model
#from tensorflow.keras.utils.vis_utils import plot_model #updated
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import tensorflow.keras.optimizers as optimizer
from sklearn import preprocessing
import data_preprocessing_cmapps as cdp
import data_preprocessing_bearing as bdp
import data_preprocessing_battery as badp
import VAE
from tensorflow.keras import layers
from tensorflow.keras import activations
from sklearn.metrics import r2_score
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
import random

#Hyperparameter 
use_VAE = False
sequence_length = 24
epochen = 300

lr_rul = 0.001
batch_rul = 32

l2_reg = True

#make sure, that train and test data are equally scaled especially RUL data
dataset_n = 'FD004'
dataset_lists = ['Bearing']
#VAE Hyperparameter
lr_vae = 0.0004
epochen_VAE = 75
intermediate_dim = 240
latent_dim=(3,80) #how should the latent space be reinterpreted as a sequence? 
latent_length = int(latent_dim[0]*latent_dim[1]) #l(flatten), this is the output dimension of the encoder
flat_rul = True
l2_reg = 0.00001

layer_list = [[3,[80,40,20]],[2,[80,20]],[4,[80,60,40,20]],[4,[100,60,40,20]],[6,[80,60,40,40,30,20]],[6,[100,80,60,60,40,40,20]],[2,[80,20]],[3,[80,40,20]],[5,[100,80,60,40,20]],[2,[30,20]]]
scheduler_start = 100
early_stop_patience = 50

#CMAPSS
def cmapss(seq_length, data_n):

    
    train_data, train_label, val_data,val_label, test_data, test_label = cdp.preprocessing_cmapps(dataset_n=data_n, seq_length = seq_length)
    
    
    train_label = train_label[:,sequence_length-1]
    val_label = val_label[:,sequence_length-1]
    test_label = test_label[:,sequence_length-1]

    max_value_rul = np.amax(np.array([np.amax(train_label),np.amax(val_label),np.amax(test_label)]))    

    #delete cylce column
    train_data = np.delete(train_data, 3, axis=2) 
    val_data = np.delete(val_data, 3, axis=2)
    test_data = np.delete(test_data, 3, axis=2)
    
    #delete setting column
    
    train_data = np.delete(train_data, 2, axis=2) 
    val_data = np.delete(val_data, 2, axis=2)
    test_data = np.delete(test_data, 2, axis=2)
    
    train_data = np.delete(train_data, 5, axis=2) 
    val_data = np.delete(val_data, 5, axis=2)
    test_data = np.delete(test_data, 5, axis=2)
    
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
    
    '''
    #offset label
    offset_label = 0.25
    train_label = train_label+offset_label
    val_label = val_label+offset_label
    test_label = test_label+offset_label
    '''
    #ad dummy dimension
    train_label = np.reshape(train_label,[train_label.shape[0],1])
    val_label = np.reshape(val_label,[val_label.shape[0],1])
    test_label = np.reshape(test_label,[test_label.shape[0],1])
    
    return train_data, train_label, val_data,val_label, test_data, test_label, max_value_rul


def battery(seq_length):
    train_data, train_label, val_data,val_label, test_data, test_label, max_value_rul = badp.preprocessing_battery(seq_length)

#bearing dataset
def bearing(seq_length, log, flat = True):
    train_data, train_label, val_data,val_label, test_data, test_label, max_value_rul = bdp.preprocessing_bearing(seq_length, log)
    
    train_label = train_label[:,seq_length-1,:]
    val_label = val_label[:,seq_length-1,:]
    test_label = test_label[:,seq_length-1,:]
    
    
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
    #drop cycle col
    train_data = train_data[:,:,1:train_data.shape[2]]
    val_data = val_data[:,:,1:val_data.shape[2]]
    test_data = test_data[:,:,1:test_data.shape[2]]
    
    #maybe drop position col aswell
    
    return train_data, train_label, val_data,val_label, test_data, test_label, max_value_rul

def battery(seq_length):
    return badp.preprocessing_battery(seq_length)


# bearing train_data, train_label, val_data,val_label, test_data, test_label, max_value_rul = bearing(sequence_length, False, flat = True)
# battery train_data, train_label, val_data,val_label, test_data, test_label, max_value_rul = battery(sequence_length)



def r2_keras(y_true, y_pred):
   
    res =  K.sum(K.square( y_true - y_pred ))
    tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - res/(tot + K.epsilon()) )

def create_model(nlayer, units, input_shape):

    opt = optimizer.Adam(learning_rate=0.001,beta_2=0.90)
    
    
    # Next, we build a deep network. 
    # The first layer ==> LSTM layer with 100 units
    # Second Layer ==> Convolution Layer
    # Third Layer ==>  LSTM layer with 50 units. 
    # Dropout is applied after each LSTM layer to control overfitting. 
    # Final layer is a Dense output layer with single unit and Relu activation.


    
    model = Sequential()
    model.add(InputLayer(input_shape=(input_shape[0], input_shape[1])))
    for i in range(nlayer): 
        if (i == 0) and l2_reg: # or i==1
            model.add((GRU(units=units[i],return_sequences=True,kernel_regularizer=tf.keras.regularizers.L2(l2=l2_reg))))
            if units[i] > 30:
                model.add(Dropout(0.2)) 
            else:
                model.add(Dropout(0.1)) 
        else:
            if i == nlayer-1: #last layer => return_sequence = False
                model.add((GRU(units[i],return_sequences=False)))
                if units[i] > 30:
                    model.add(Dropout(0.2)) 
                else:
                    model.add(Dropout(0.1)) 
            else:
                model.add((GRU(units[i],return_sequences=True)))
                if units[i] > 30:
                    model.add(Dropout(0.2)) 
                else:
                    model.add(Dropout(0.1)) 
    
    model.add(Dense(units = 20))
    #model.add(Dense(units = 10))
    #model.add(Dense(units = 5))
    model.add(Dense(units=1))
    model.add(layers.Activation(activations.linear))
    model.compile(loss='mse', optimizer=opt,metrics=['mae'])
    
    print(model.summary())
    
    return model

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
        
        
ITERATIONS = 10

results = pd.DataFrame(columns=['MSE val', 'std_MSE val', '`MSE train', 'std_MSE train',   # bigger std means less robust
                                'epochs', 'number of layer', 'units of layer', 
                                'dropout', 'batch_size', 
                                'sequence_length', 'dataset','regu','patience','l2','flat rul', 'lr'])  

weights_file = 'lstm_hyper_parameter_weights.h5'

alpha_list = [0.01, 0.05] + list(np.arange(10,60+1,10)/100)

sequence_list = list(np.arange(10,40+1,5))
epoch_list = list(np.arange(5,20+1,5))
nodes_list = [[32], [64], [128], [256], [32, 64], [64, 128], [128, 256]]

# lowest dropout=0.1, because I know zero dropout will yield better training results but worse generalization


# again, earlier testing revealed relu performed significantly worse, so I removed it from the options
activation_functions = ['tanh', 'sigmoid']
batch_size_list = [32, 64, 128, 256]
sensor_list = [['s_2', 's_3', 's_4', 's_7', 's_8', 's_9', 's_11', 's_12', 's_13', 's_14', 's_15', 's_17', 's_20', 's_21'],
               ['s_2', 's_3', 's_4', 's_7', 's_11', 's_12', 's_15', 's_17', 's_20', 's_21']]

tuning_options = np.prod([len(alpha_list),
                          len(sequence_list),
                          len(epoch_list),
                          len(nodes_list),
                          #len(dropouts),
                          len(activation_functions),
                          len(batch_size_list),
                          len(sensor_list)])
#später beide layer listen zusammen fassen und für andere sequenz längen testen


for datan in dataset_lists:
    #train_data, train_label, val_data,val_label, test_data, test_label, max_value_rul = cmapss(sequence_length,datan)
    train_data, train_label, val_data,val_label, test_data, test_label, max_value_rul  = bearing(sequence_length, False)
    for i in range(len(layer_list)):
    
        
        mse_list_val = []
        mse_list_train = []
        early_stop = []
        '''
        # init parameters
        alpha = random.sample(alpha_list, 1)[0]
        sequence_length = random.sample(sequence_list, 1)[0]
        epochs = random.sample(epoch_list, 1)[0]
        nodes_per_layer = random.sample(nodes_list, 1)[0]
        dropout = random.sample(dropouts, 1)[0]
        activation = random.sample(activation_functions, 1)[0]
        batch_size = random.sample(batch_size_list, 1)[0]
        remaining_sensors = random.sample(sensor_list, 1)[0]
        layer = random.sample(layer_list) #ex: ([5],[80,60,40,30,20])
        '''
        sequence_length = 24
        layer = layer_list[i]
        
        
        
        
        input_shape = (sequence_length, train_data.shape[2])
        
        def scheduler(epoch,lr):
            if epoch < scheduler_start:
                return lr
            else:
                return lr * tf.math.exp(-0.04)
        
        my_callbacks = [
                tf.keras.callbacks.EarlyStopping(monitor="val_loss",verbose=1, restore_best_weights=True, patience=early_stop_patience),
                tf.keras.callbacks.LearningRateScheduler(scheduler)
        ]
        
        
    
        for repeat in range(ITERATIONS): #wie oft soll das selbe setup trainiert werden?
            model = create_model(layer[0], layer[1], input_shape)
            # Define learning rate
            K.set_value(model.optimizer.learning_rate, lr_rul)
            # fit the network
            history = model.fit(train_data, train_label, epochs=epochen, batch_size=batch_rul, shuffle=False,validation_data=(val_data,val_label), verbose=2,
                      callbacks = my_callbacks)
            #[keras.callbacks.ModelCheckpoint(model_path,monitor='val_loss', save_best_only=True, mode='min', verbose=0)]
            
    
            
            def mse(y_pred,y_true):
                return np.sum(np.square(y_pred-y_true))/y_pred.shape[0]
            
            def mae(y_pred,y_true):
                return np.sum(np.abs((y_pred-y_true)))/y_pred.shape[0]
            
            
            
            if True:
                #test metrics
            
                y_pred_test = model.predict(val_data)
                y_test_test = model.predict(test_data)
                y_true_test = val_label
                    
                val_mse = mean_squared_error(y_pred_test, y_true_test)
                train_mse = history.history['loss'][-1]
                val_mae = mae(y_pred_test,y_true_test)
                print('val_mse: ' + str(val_mse))
                print('val_mae: ' + str(val_mae)) 
                      
                print('test_mse: ' + str(mse(y_test_test,test_label)))
                print('test_mae: ' + str(mae(y_test_test,test_label)))        
                print('test_r^2: ' + str(r2_score(y_test_test, test_label)))
                
                #uncomment if plots are requiered
                fig_verify = plt.figure(figsize=(10, 5))
                plt.plot(y_pred_test[0:1000]*max_value_rul, color="blue")
                plt.plot(val_label[0:1000]*max_value_rul, color="green")
                plt.title('Predicted and real RUL (val) id: ' + str(i))
                plt.ylabel('RUL')
                plt.xlabel('row')
                plt.legend(['predicted', 'actual data'], loc='upper left')
                plt.show()
    
                # Plot in blue color the predicted data and in green color the
                # actual data to verify visually the accuracy of the model.
                fig_verify = plt.figure(figsize=(10, 5))
                plt.plot(y_test_test[0:3000]*max_value_rul, color="blue")
                plt.plot(test_label[0:3000]*max_value_rul, color="green")
                plt.title('Predicted and real RUL (test) id: ' + str(i))
                plt.ylabel('RUL')
                plt.xlabel('row')
                plt.legend(['predicted', 'actual data'], loc='upper left')
                plt.show()
                if val_mse < 0.09:
                    mse_list_val.append(val_mse)
                    mse_list_train.append(history.history['loss'][-1])
                    early_stop.append(len(history.history['loss'])-early_stop_patience)
        #epochs model.history 
        # append results, andere parameter anpassen
        try:
            d = {'MSE val':np.mean(mse_list_val), 'std_MSE val':np.std(mse_list_val), 
                 'MSE train':np.mean(mse_list_train), 'std_MSE train':np.std(mse_list_train), 
                 'epochs':np.mean(early_stop), 'number of layer':str(layer[0]), 'units of layer':str(layer[1]), 
                 'batch_size':batch_rul, 'sequence_length':sequence_length, 'dataset':dataset_n, 'regu':l2_reg, 'patience':early_stop_patience,
                 'l2':str(l2_reg),'flat rul':str(flat_rul), 'lr': str(lr_rul)
            }
        
            results = results.append(pd.DataFrame(d, index=[0]), ignore_index=True)
        
            results.to_csv('rnn_rul_hyperparameter_flat_line_nol2.csv', sep=';')
        except:
            #dann waren es nur flat line predictions
            pass
    
