# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 11:23:55 2022

@author: POV GmbH
"""
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Activation, Dense, Input, Dropout, RepeatVector, MaxPooling1D
from tensorflow.keras.layers import Conv1D, Conv2D, Flatten, GRU, LSTM
from tensorflow.keras.layers import Reshape, Conv2DTranspose, UpSampling1D
from tensorflow.keras.layers import LeakyReLU, ReLU, Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import concatenate, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers.legacy import RMSprop, Adam
import data_preprocessing_cmapps as cdp
import data_preprocessing_battery as badp
import data_preprocessing_bearing as bdp
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd
import clustering as cl
#import tensorview as tv
"""
### This section of the code is derived from 
Yoon, J., Jarrett, D., & van der Schaar, M. (2019). 
Time-series Generative Adversarial Networks. 
Advances in Neural Information Processing Systems, 5509–5519.
you can find my fork of their code here: https://github.com/firmai/tsgan

"""
#Label for Discriminator real = 1 and fake = 0
epochs_enc = 100
patience_encoder = 40
seq_length = 24 #specifying the lenght of series
dataset_n ='FD001'
flat_rul = True
steps = 1500

def cmapss(seq_length, dataset_n):

    
    train_data, train_label, val_data,val_label, test_data, test_label = cdp.preprocessing_cmapps(dataset_n=dataset_n, seq_length = seq_length)
    
    '''
    train_label = train_label[:,sequence_length-1]
    val_label = val_label[:,sequence_length-1]
    test_label = test_label[:,sequence_length-1]
    '''
    max_value_rul = np.amax(np.array([np.amax(train_label),np.amax(val_label),np.amax(test_label)]))    
    
    '''
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
    '''
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
    
    
    #ad dummy dimension
    train_label = np.reshape(train_label,[train_label.shape[0],train_label.shape[1],1])
    val_label = np.reshape(val_label,[val_label.shape[0],val_label.shape[1],1])
    test_label = np.reshape(test_label,[test_label.shape[0],test_label.shape[1],1])
    
    return train_data, train_label, val_data,val_label, test_data, test_label, max_value_rul

#bearing dataset
def bearing(seq_length, log, freq_gab, flat = True):
    new = False
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
    
    return train_data, train_label, val_data,val_label, test_data, test_label, max_value_rul


def battery(seq_length):
    train_data, train_label, val_data,val_label, test_data, test_label, max_value_rul = badp.preprocessing_battery(seq_length)
    train_label = np.repeat(train_label, seq_length, axis=1)
    train_label = np.expand_dims(train_label,axis=2)
    val_label = np.repeat(val_label, seq_length, axis=1)
    val_label = np.expand_dims(val_label,axis=2)
    test_label = np.repeat(test_label, seq_length, axis=1)
    test_label = np.expand_dims(test_label,axis=2)
    
    return train_data, train_label, val_data,val_label, test_data, test_label, max_value_rul

train_data, train_label, val_data,val_label, test_data, test_label, max_value_rul = bearing(seq_length, False, 120, flat = True)
#train_data, train_label, val_data,val_label, test_data, test_label, max_value_rul = battery(seq_length)
#train_data, train_label, val_data,val_label, test_data, test_label, max_value_rul = cmapss(seq_length,dataset_n)
Z_dim = 50 #size of latent vector
#gen_concat = np.concatenate([train_data[:,:,3:4],train_label],axis=2)
features_n = train_data.shape[2] #RUL column will be later added
num_labels = 11 
num_clusters = 12
noise_dim = seq_length*features_n
SHAPE = (seq_length, features_n) #Shape der Daten muss noch unterschei
features_n_c = 10 #number of cluster

hidden_dim = features_n*4
cond_col = 4 #How many columns does the condition vector have
cond_length = cond_col+features_n_c #Summe aus RUL, Cycle und Cluster Information
#%% PCA Analysis
    
def PCA_Analysis (dataX, dataX_hat):
  
    # Analysis Data Size
    Sample_No = min(len(dataX)-100, len(dataX_hat)-100)
    
    # Data Preprocessing
    for i in range(Sample_No):
        if (i == 0):
            arrayX = np.reshape(np.mean(np.asarray(dataX[0]),1), [1,len(dataX[0][:,0])])
            arrayX_hat = np.reshape(np.mean(np.asarray(dataX_hat[0]),1), [1,len(dataX[0][:,0])])
        else:
            arrayX = np.concatenate((arrayX, np.reshape(np.mean(np.asarray(dataX[i]),1), [1,len(dataX[0][:,0])])))
            arrayX_hat = np.concatenate((arrayX_hat, np.reshape(np.mean(np.asarray(dataX_hat[i]),1), [1,len(dataX[0][:,0])])))
    
    # Parameters        
    No = len(arrayX[:,0])
    colors = ["red" for i in range(No)] +  ["blue" for i in range(No)]    
    
    # PCA Analysis
    pca = PCA(n_components = 2)
    pca.fit(arrayX)
    pca_results = pca.transform(arrayX)
    pca_hat_results = pca.transform(arrayX_hat)
        
    # Plotting
    f, ax = plt.subplots(1)
    
    plt.scatter(pca_results[:,0], pca_results[:,1], c = colors[:No], alpha = 0.2, label = "Original")
    plt.scatter(pca_hat_results[:,0], pca_hat_results[:,1], c = colors[No:], alpha = 0.2, label = "Synthetic")

    ax.legend()
    
    plt.title('PCA plot')
    plt.xlabel('x-pca')
    plt.ylabel('y_pca')
    plt.show()
    
    
#%% TSNE Analysis
    
def tSNE_Analysis (dataX, dataX_hat):
  
    # Analysis Data Size
    Sample_No = min(len(dataX)-100, len(dataX_hat)-100)
  
    # Preprocess
    for i in range(Sample_No):
        if (i == 0):
            arrayX = np.reshape(np.mean(np.asarray(dataX[0]),1), [1,len(dataX[0][:,0])])
            arrayX_hat = np.reshape(np.mean(np.asarray(dataX_hat[0]),1), [1,len(dataX[0][:,0])])
        else:
            arrayX = np.concatenate((arrayX, np.reshape(np.mean(np.asarray(dataX[i]),1), [1,len(dataX[0][:,0])])))
            arrayX_hat = np.concatenate((arrayX_hat, np.reshape(np.mean(np.asarray(dataX_hat[i]),1), [1,len(dataX[0][:,0])])))
     
    # Do t-SNE Analysis together       
    final_arrayX = np.concatenate((arrayX, arrayX_hat), axis = 0)
    
    # Parameters
    No = len(arrayX[:,0])
    colors = ["red" for i in range(No)] +  ["blue" for i in range(No)]    
    
    # TSNE anlaysis
    tsne = TSNE(n_components = 2, verbose = 1, perplexity = 40, n_iter = 300)
    tsne_results = tsne.fit_transform(final_arrayX)
    
    # Plotting
    f, ax = plt.subplots(1)
    
    plt.scatter(tsne_results[:No,0], tsne_results[:No,1], c = colors[:No], alpha = 0.2, label = "Original")
    plt.scatter(tsne_results[No:,0], tsne_results[No:,1], c = colors[No:], alpha = 0.2, label = "Synthetic")

    ax.legend()
    
    plt.title('t-SNE plot')
    plt.xlabel('x-tsne')
    plt.ylabel('y_tsne')
    plt.show()


def MinMax(train_data):
  scaler = MinMaxScaler()
  num_instances, num_time_steps, num_features = train_data.shape
  train_data = np.reshape(train_data, (-1, num_features))
  train_data = scaler.fit_transform(train_data)
  train_data = np.reshape(train_data, (num_instances, num_time_steps, num_features))
  return train_data, scaler



def end_cond(X_train):
  vals = X_train[:,23,4]/X_train[:,0,4]-1

  comb1 = np.where(vals<-.1,0,0)
  comb2 = np.where((vals>=-.1)&(vals<=-.05),1,0)
  comb3 = np.where((vals>=-.05)&(vals<=-.0),2,0)
  comb4 = np.where((vals>0)&(vals<=0.05),3,0)
  comb5 = np.where((vals>0.05)&(vals<=0.1),4,0)
  comb6 = np.where(vals>0.1,5,0)
  cond_all = comb1 + comb2 + comb3+ comb4+ comb5+ comb6

  print(np.unique(cond_all, return_counts=True))
  arr = np.repeat(cond_all,24, axis=0).reshape(len(cond_all),24)
  X_train = np.dstack((X_train, arr))
  return X_train
    




#dataX, X_train, scaler, all_scalar = dp.preprocessing_cmapps(sl=seq_length, short=False)


#noise_dim = seq_length*features_n
#SHAPE = (seq_length, features_n)
#hidden_dim = features_n*4

def generator(inputs,
              activation='relu',
              lat=None,
              cond=None):


    if cond is not None:
        # generator 0 of MTSS
        #inputs, conv = inputs
        inputs = [inputs, lat, cond]
        x = concatenate(inputs, axis=1)
        # noise inputs + conditional codes
        #x = inputs #inputs has the output of generator 0 and will be kept seperate with the condition vector
    else:
        # default input is just a noise dimension (z-code)
        x = inputs ## 
    
    x = Dense(SHAPE[0]*SHAPE[1])(x)
    x = Reshape((SHAPE[0], SHAPE[1]))(x)
    #x = LSTM(480, return_sequences=True, return_state=False,unroll=True)(x)
    #x = LSTM(72, return_sequences=True, return_state=False,unroll=True)(x)

    #x = Reshape((int(SHAPE[0]/2), 6))(x)
    x = Conv1D(filters=features_n, kernel_size=2, strides=1,padding= "same")(x)
    x = MaxPooling1D(pool_size=2,strides=1, padding='same')(x)
    x = Conv1D(filters=features_n, kernel_size=2, strides=1,padding= "same")(x)
    x = MaxPooling1D(pool_size=2,strides=1, padding='same')(x)
    x = Conv1D(filters=features_n, kernel_size=2, strides=1,padding= "same")(x)
    x = MaxPooling1D(pool_size=2,strides=1, padding='same')(x)
    #x = BatchNormalization(momentum=0.8)(x) # adjusting and scaling the activations
    x = LSTM(int(features_n*4), return_sequences=True)(x)
    x = LSTM(int(features_n*2), return_sequences=True)(x)
    x = LSTM(int(features_n), return_sequences=True)(x)
    x = Dense(features_n)(x)
    #x = BatchNormalization(momentum=0.8)(x)
    '''
    #x = LSTM(480, return_sequences=True, return_state=False,unroll=True)(x)
    #x = LSTM(72, return_sequences=True, return_state=False,unroll=True)(x)
    x = LSTM(72, return_sequences=True, return_state=False,unroll=True)(x)
    x = LSTM(72, return_sequences=False, return_state=False,unroll=True)(x)
    x = Reshape((int(SHAPE[0]/2), 6))(x)
    x = Conv1D(128, 4, 1, "same")(x)
    x = BatchNormalization(momentum=0.8)(x) # adjusting and scaling the activations
    x = ReLU()(x)
    x = UpSampling1D()(x)
    x = Conv1D(features_n, 4, 1, "same")(x) 
    x = BatchNormalization(momentum=0.8)(x)
    '''
    
    if activation is not None:
        x = Activation(activation)(x)
    
    # generator output is the synthesized data x
    return Model(inputs, x,  name='gen1')

def discriminator(inputs,
                  activation='sigmoid',
                  num_labels=None,
                  num_codes=None):

    #ints = int(SHAPE[0]/2)
    x = inputs

    x = LSTM(units=100, return_sequences=(True))(x)
    x = Dropout(0.2)(x)
    x = LSTM(units=75, return_sequences=(True))(x)
    x = Dropout(0.2)(x)
    x = LSTM(units=50, return_sequences=(False))(x)
    x = Dropout(0.1)(x)
    x = Dense(20)(x)
    '''
    x = GRU(SHAPE[1]*SHAPE[0] , return_sequences=False, return_state=False,unroll=True, activation="relu")(x)
    x = Reshape((SHAPE[0], SHAPE[1]))(x)
    x = Conv1D(16, 3,2, "same")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv1D(32, 3, 2, "same")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv1D(64, 3, 2, "same")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv1D(128, 3, 1, "same")(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Flatten()(x)
    '''
    # default output is probability that the time series array is real
    outputs = Dense(1)(x)

    if num_codes is not None:
        # MTSS-GAN Q0 output
        # z0_recon is reconstruction of z0 normal distribution
        # eventually two loss functions from this output.
        z0_recon =  Dense(num_codes)(x)
        z0_recon = Activation('tanh', name='z0')(z0_recon)
        outputs = [outputs, z0_recon]

    return Model(inputs, outputs, name='discriminator')
'''
def discriminator(inputs,
                  activation='sigmoid',
                  num_labels=None,
                  num_codes=None):

    #ints = int(SHAPE[0]/2)
    x = inputs
    x = GRU(SHAPE[1]*SHAPE[0] , return_sequences=False, return_state=False,unroll=True, activation="relu")(x)
    x = Reshape((SHAPE[0], SHAPE[1]))(x)
    x = Conv1D(16, 3,2, "same")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv1D(32, 3, 2, "same")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv1D(64, 3, 2, "same")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv1D(128, 3, 1, "same")(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Flatten()(x)
    # default output is probability that the time series array is real
    outputs = Dense(1)(x)

    if num_codes is not None:
        # MTSS-GAN Q0 output
        # z0_recon is reconstruction of z0 normal distribution
        # eventually two loss functions from this output.
        z0_recon =  Dense(num_codes)(x)
        z0_recon = Activation('tanh', name='z0')(z0_recon)
        outputs = [outputs, z0_recon]

    return Model(inputs, outputs, name='discriminator')
'''
def build_encoder(inputs, cond_shape,feature0_dim): 
    
    #feature0_dim = int((train_data.shape[1]*train_data.shape[2]-cond_shape)/2)
    
    x, feature0 = inputs

    y = GRU(SHAPE[0]*SHAPE[1], return_sequences=False, return_state=False,unroll=True)(x)
    y = Flatten()(y)
    y = Dense(units = SHAPE[0]*SHAPE[1])(y)
    y = Dense(units=feature0_dim+int(feature0_dim/2))(y)
    y = Dense(units=feature0_dim, activation='relu')(y)
    # Encoder0 or enc0: data to feature0 
    enc0 = Model(inputs=x, outputs=y, name="encoder0")
    
    # Encoder1 or enc1
    
    y = Dense(cond_shape)(feature0)
    labels = Activation('softmax')(y)
    # Encoder1 or enc1: feature0 to class labels 
    enc1 = Model(inputs=feature0, outputs=labels, name="encoder1")

    # return both enc0 and enc1
    return enc0, enc1

def build_generator(latent_codes,feature0_dim):
    """Build Generator Model sub networks
    Two sub networks: 1) Class and noise to feature0 
        (intermediate feature)
        2) feature0 to time series array
    # Arguments
        latent_codes (Layers): dicrete code (labels),
            noise and feature0 features
        feature0_dim (int): feature0 dimensionality
    # Returns
        gen0, gen1 (Models): Description below
    """
    
    # Latent codes and network parameters
    labels, z0, z1, cond,feature0 = latent_codes

    # gen0 inputs
    inputs = [labels, z0]      # 58 + 128 = 62-dim
    x = concatenate(inputs, axis=1)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = RepeatVector(seq_length)(x)
    x = LSTM(units=128, return_sequences=True)(x)
    x = LSTM(units= feature0_dim, return_sequences=False)(x)
    fake_feature0 = Dense(feature0_dim, activation='relu')(x)

    # gen0: classes and noise (labels + z0) to feature0
    gen0 = Model(inputs, fake_feature0, name='gen0')
    print('feat0')
    print(feature0.shape)
    # gen1: feature0 + z0 to feature1 (time series array)
    # example: Model([feature0, z0], (steps, feats),  name='gen1')
    gen1 = generator(feature0, lat=z1, cond=cond)#, labels= con_vec)

    return gen0, gen1


def build_discriminator(inputs, z_dim=50):
    """Build Discriminator 1 Model
    Classifies feature0 (features) as real/fake time series array and recovers
    the input noise or latent code (by minimizing entropy loss)
    # Arguments
        inputs (Layer): feature0
        z_dim (int): noise dimensionality
    # Returns
        dis0 (Model): feature0 as real/fake recovered latent code
    """

    # input is 256-dim feature1
    x = Dense(SHAPE[0]*SHAPE[1], activation='relu')(inputs)
  
    x = Dense(SHAPE[0]*SHAPE[1], activation='relu')(x)


    # first output is probability that feature0 is real
    f0_source = Dense(1)(x)
    f0_source = Activation('sigmoid',
                           name='feature1_source')(f0_source)

    # z0 reonstruction (Q0 network)
    z0_recon = Dense(z_dim)(x) 
    z0_recon = Activation('tanh', name='z0')(z0_recon)
    
    discriminator_outputs = [f0_source, z0_recon]
    dis0 = Model(inputs, discriminator_outputs, name='dis0')
    return dis0

def train_encoder(model,
                  data, 
                  model_name="MTSS-GAN", 
                  batch_size=64):
    """ Train the Encoder Model (enc0 and enc1)
    # Arguments
        model (Model): Encoder
        data (tensor): Train and test data
        model_name (string): model name
        batch_size (int): Train batch size
    """

    (x_train, y_train), (x_test, y_test) = data
    my_callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss",verbose=1, restore_best_weights=True, patience=patience_encoder),
                    #keras.callbacks.ModelCheckpoint(model_path,monitor='val_loss', save_best_only=True, mode='min', verbose=0)
                    ]
    model.compile(loss='mse',
                  optimizer='adam',
                  metrics=['mse'])
    model.fit(x_train,
              y_train[:,0:num_labels], #enoceder versucht die timeseries sequences dem cluster zuzuordnen. Aber warum? dafür ist doch der cluster al
              validation_data=(x_test, y_test[:,0:num_labels]),
              epochs=epochs_enc,
              batch_size=batch_size,
              callbacks = my_callbacks)
    #warum nur 10 epochen für den encoder?
    model.save(model_name + "-encoder.h5")
    score = model.evaluate(x_test,
                           y_test[:,0:num_clusters], 
                           batch_size=batch_size,
                           verbose=0)
    print("\nTest accuracy: %.1f%%" % (100.0 * score[1]))

def build_and_train_models(train_steps = 2000):
    """Load the dataset, build MTSS discriminator,
    generator, and adversarial models.
    Call the MTSS train routine.
    """

    train_n = int((train_data.shape[0])*.95)
    #X = data[:,:,:-1]
    #y = data[:,-1,-1]
    
    #calculate clusters
    #X = np.concatenate([train_data,train_label], axis=2) #put the RUL Label into the trainingdata
    X = train_data
    clusters = cl.birch_clustering(np.concatenate([train_data,train_label],axis=2),features_n=features_n_c)
    #add dummy dimensions
    #clusters = np.reshape(clusters, [clusters.shape[0],1,1])
    #clusters = np.repeat(clusters, seq_length, axis =1) #a cluster is calculated for one sequence, so it must be repeated for the sequence length, to fit the condition vector
    #cycle = X[:,:,2:4] #get third column of seq array, while maintaining shape
    #col7 = X[:,:,7:8]
    
    #Condtion_vector = np.concatenate([cycle,col7,train_label,clusters], axis = 2)
    
    
    #y = clusters[:,0,0] #get the first element out of a sequence, which is only needed for the label vector
    #label = train_label[:,seq_length-1,:]
    #cycle = train_data[:,seq_length-1,3:4]
    #X = np.concatenate([X,train_label],axis=2)
    #label = np.concatenate([cycle,col7,train_label],axis=2) #col7
    #del clusters
    #split in Train and Testdata, also split in data x and label y
    y = to_categorical(clusters)
    #label = label.reshape([label.shape[0],label.shape[1]*label.shape[2]]) #flatten 2nd and 3rd dimension
    
    y = np.concatenate([train_label[:,seq_length-1,:],y],axis= 1)
    
    
    
    indices = tf.range(start=0, limit=tf.shape(train_data)[0], dtype=tf.int32)
    shuffled_indices = tf.random.shuffle(indices)
    
    #vor dem validation split, mische die Sequenzen, damit im validation split nicht nur die letzten Sequenzen enthalten sind
    y = tf.gather(y, shuffled_indices)
    X = tf.gather(X, shuffled_indices)
    
    x_train, y_train = X[:train_n,:,:], y[:train_n]
    x_test, y_test = X[train_n:,:,:], y[train_n:]

    cond_shape = y.shape[1]

    model_name = "MTSS-GAN"
    # network parameters
    batch_size = 128

    #train_steps = 2000

    #lr = 2e-4 #original
    lr = 0.00001 #like cgan
    decay = 6e-8
    z_dim = Z_dim ##this is the  noise vector dimension, original 50
    z_shape = (z_dim, )
    feature0_dim = int((train_data.shape[1]*train_data.shape[2]-cond_shape)/2)#SHAPE[0]*SHAPE[1]
    feature0_shape = (feature0_dim, )
    # [1] uses Adam, but discriminator converges easily with RMSprop
    optimizer = Adam(lr=lr)#RMSprop(lr=lr, decay=decay)

    # build discriminator 0 and Q network 0 models
    input_shape = (feature0_dim, )
    inputs = Input(shape=input_shape, name='discriminator0_input')
    dis0 = build_discriminator(inputs, z_dim=z_dim )
    #Model(Dense(SHAPE[0]*SHAPE[1]), [f0_source, z0_recon], name='dis0')

    # loss fuctions: 1) probability feature0 is real 
    # (adversarial0 loss)
    # 2) MSE z0 recon loss (Q0 network loss or entropy0 loss)
    # Because there are two outputs. 

    loss = ['binary_crossentropy', 'mse']
    loss_weights = [1.0, 1.0] 
    dis0.compile(loss=loss,
                 loss_weights=loss_weights,
                 optimizer=optimizer,
                 metrics=['accuracy'])
    dis0.summary() # feature0 discriminator, z0 estimator

    # build discriminator 1 and Q network 1 models

    input_shape = (x_train.shape[1], x_train.shape[2])
    inputs = Input(shape=input_shape, name='discriminator1_input')
    dis1 = discriminator(inputs, num_codes=z_dim)

    # loss fuctions: 1) probability time series arrays is real (adversarial1 loss)
    # 2) MSE z1 recon loss (Q1 network loss or entropy1 loss)
    loss = ['binary_crossentropy', 'mse']
    loss_weights = [1.0, 10.0] 
    dis1.compile(loss=loss,
                 loss_weights=loss_weights,
                 optimizer=optimizer,
                 metrics=['accuracy'])
    dis1.summary() # time series array discriminator, z1 estimator 


    # build generator models
    label_shape = (cond_shape, )
    feature0 = Input(shape=feature0_shape, name='feature0_input')
    labels = Input(shape=label_shape, name='labels')
    z0 = Input(shape=z_shape, name="z0_input")
    z1 = Input(shape=z_shape, name="z1_input")
    cond = Input(shape=label_shape, name="cond_input")
    #conv_vec =Input(shape=(seq_length,2), name="condition_vector")
    #replace latent vector z1 with conditionvector for gen1
    latent_codes = (labels, z0, z1, cond, feature0)
    feature0_dim = int((train_data.shape[1]*train_data.shape[2]-cond_shape)/2)
    
    gen0, gen1 = build_generator(latent_codes, feature0_dim)
    # gen0: classes and noise (labels + z0) to feature0 
    gen0.summary() # (latent features generator)
    # gen1: feature0 + z0 to feature1 
    gen1.summary() # (time series array generator )

    # build encoder models
    input_shape = SHAPE
    inputs = Input(shape=input_shape, name='encoder_input')
    enc0, enc1 = build_encoder((inputs, feature0), cond_shape, feature0_dim)
     # Encoder0 or enc0: data to feature0  
    enc0.summary() # time series array to feature0 encoder
     # Encoder1 or enc1: feature0 to class labels
    enc1.summary() # feature0 to labels encoder (classifier)
    encoder = Model(inputs, enc1(enc0(inputs)), name='Encoder')
    encoder.summary() # time series array to labels encoder (classifier)

    data = (x_train, y_train), (x_test, y_test)


    # this process would train enco, enc1, and encoder
    train_encoder(encoder, data, model_name=model_name)


    # build adversarial0 model = 
    # generator0 + discriminator0 + encoder1
    # encoder0 weights frozen
    enc1.trainable = False
    # discriminator0 weights frozen
    dis0.trainable = False
    gen0_inputs = [labels, z0]
    gen0_outputs = gen0(gen0_inputs)
    adv0_outputs = dis0(gen0_outputs) + [enc1(gen0_outputs)]
    # labels + z0 to prob labels are real + z0 recon + feature1 recon
    adv0 = Model(gen0_inputs, adv0_outputs, name="adv0")
    # loss functions: 1) prob labels are real (adversarial1 loss)
    # 2) Q network 0 loss (entropy0 loss)
    # 3) conditional0 loss (classifier error)
    loss_weights = [1.0, 1.0, 1.0] 
    loss = ['binary_crossentropy', 
            'mse',
            'categorical_crossentropy']
    adv0.compile(loss=loss,
                 loss_weights=loss_weights,
                 optimizer=optimizer,
                 metrics=['accuracy'])
    adv0.summary()

    # build adversarial1 model =
    # generator1 + discriminator1 + encoder0
    optimizer = RMSprop(lr=lr*0.5, decay=decay*0.5)
    # encoder1 weights frozen
    enc0.trainable = False
    # discriminator1 weights frozen
    dis1.trainable = False
    gen1_inputs = [feature0,z1, cond]
    gen1_outputs = gen1(gen1_inputs)
    '''
    cond_r = cond[:,10:58] #remove cluster information

    cond_r = tf.cast(cond_r, tf.float32)
    print(cond_r.dtype)
    #cond_r = np.asarray(cond_r)
    #cond_r = np.reshape(cond_r,[cond_r.shape[0],24,2])
    cond_r = tf.convert_to_tensor(cond_r)
    cond_r = tf.reshape(cond_r,(cond.shape[0],seq_length,int(y.shape[1]/seq_length))) #reshape rul and cycle information into two columns
    gen1_outputs = tf.concat([gen1_outputs[:,:,0:3],cond_r[:,:,0:1],gen1_outputs[:,:,3:18],cond_r[:,:,1:2]],axis=2)
    
    #gen1_outputs verktten mit label vektor???
    '''
    
    adv1_outputs = dis1(gen1_outputs) + [enc0(gen1_outputs)]
    # feature1 + z1 to prob feature1 is 
    # real + z1 recon + feature1/time series array recon
    adv1 = Model(gen1_inputs, adv1_outputs, name="adv1")
    # loss functions: 1) prob feature1 is real (adversarial0 loss)
    # 2) Q network 1 loss (entropy1 loss)
    # 3) conditional1 loss
    loss = ['binary_crossentropy', 'mse', 'mse']
    loss_weights = [1.0, 10.0, 1.0] 
    adv1.compile(loss=loss,
                 loss_weights=loss_weights,
                 optimizer=optimizer,
                 metrics=['accuracy'])
    adv1.summary()

    

    # train discriminator and adversarial networks
    models = (enc0, enc1, gen0, gen1, dis0, dis1, adv0, adv1)
    params = (batch_size, train_steps, num_labels, z_dim, model_name)
    gen0, gen1, losses, acc_fake,acc_real  = train(models, data, params)


    return gen0, gen1, losses, acc_fake,acc_real 

def train(models, data, params):
    """Train the discriminator and adversarial Networks
    Alternately train discriminator and adversarial networks by batch.
    Discriminator is trained first with real and fake time series array,
    corresponding one-hot labels and latent codes.
    Adversarial is trained next with fake time series array pretending
    to be real, corresponding one-hot labels and latent codes.
    Generate sample time series data per save_interval.
    # Arguments
        models (Models): Encoder, Generator, Discriminator,
            Adversarial models
        data (tuple): x_train, y_train data
        params (tuple): Network parameters
    """
    # the MTSS-GAN and Encoder models

    enc0, enc1, gen0, gen1, dis0, dis1, adv0, adv1 = models
    # network parameters
    batch_size, train_steps, num_labels, z_dim, model_name = params
    # train dataset
    (x_train, y_train), (_, _) = data
    # the generated time series array is saved every 500 steps
    save_interval = 500
    acc_fake = []
    acc_real = []
    #überflüssig
    '''
    # label and noise codes for generator testing
    z0 = np.random.normal(scale=0.5, size=[SHAPE[0], z_dim])
    z1 = np. random.normal(scale=0.5, size=[SHAPE[0], z_dim])
    
    noise_params = [noise_class, z0, z1]
    '''
    noise_class = np.eye(num_labels)[np.arange(0, SHAPE[0]) % num_labels] #hier sollte als conditionvektor auch wie im anderen GAN die conditions genommen werden, die auch in der Real sequenz sind, und nicht monoton genierte!!

    # number of elements in train dataset
    train_size = x_train.shape[0]
    print(model_name,
          "Labels for generated time series arrays: ",
          np.argmax(noise_class, axis=1))
    
    losses = []
    
    global out_dis0
    global out_dis1

    #tv_plot = tv.train.PlotMetrics(columns=5, wait_num=5)
    for i in range(train_steps):
        # train the discriminator1 for 1 batch
        # 1 batch of real (label=1.0) and fake feature1 (label=0.0)
        # randomly pick real time series arrays from dataset
        dicta = {}
        rand_indexes = np.random.randint(0, train_size, size=batch_size)
        real_samples = tf.gather(x_train, rand_indexes)
        #real_samples = x_train[rand_indexes,:,:]
        # real feature1 from encoder0 output
        real_feature0 = enc0.predict(real_samples)

        # real labels from dataset
        #real_labels = y_train[rand_indexes]
        real_labels = tf.gather(y_train, rand_indexes)

        # generate fake feature1 using generator1 from
        # real labels and 50-dim z1 latent code
        fake_z0 = np.random.normal(scale=0.5,
                                   size=[batch_size, z_dim])


        fake_feature0 = gen0.predict([real_labels, fake_z0])

        # real + fake data
        feature0 = np.concatenate((real_feature0, fake_feature0))
        z0 = np.concatenate((fake_z0, fake_z0))

        # label 1st half as real and 2nd half as fake
        y = np.ones([2 * batch_size, 1])
        y[batch_size:, :] = 0

        # train discriminator1 to classify feature1 as 
        # real/fake and recover
        # latent code (z0). real = from encoder1, 
        # fake = from genenerator10
        # joint training using discriminator part of 
        # advserial1 loss and entropy0 loss
        metrics = dis0.train_on_batch(feature0, [y, z0])
        # log the overall loss only
        log = "%d: [dis0_loss: %f]" % (i, metrics[0])
        dicta["dis0_loss"] = metrics[0]
         
        # train the discriminator1 for 1 batch
        # 1 batch of real (label=1.0) and fake time series arrays (label=0.0)
        # generate random 50-dim z1 latent code
        fake_z1 = np.random.normal(scale=0.5, size=[batch_size, z_dim])
        #fake_z1 = con_vec
        # generate fake time series arrays from real feature1 and fake z1
        fake_samples = gen1.predict([real_feature0,fake_z1, real_labels])#, codes=real_labels)
        #with col7
        #rul_cylce = real_labels[:,features_n_c:cond_col*seq_length+features_n_c].reshape(real_labels.shape[0],seq_length,cond_col) #rewrite RUL and Cycle information in 2 columns
        #without col7
        #rul_cylce = real_labels[:,features_n_c:cond_col*seq_length+features_n_c].reshape(real_labels.shape[0],seq_length,cond_col) #rewrite RUL and Cycle information in 2 columns
        #with col7
        #fake_samples = np.concatenate([fake_samples[:,:,0:2],rul_cylce[:,:,0:2],fake_samples[:,:,4:7],rul_cylce[:,:,2:3],fake_samples[:,:,8:19],rul_cylce[:,:,3:4]],axis=2) #concate generated Data with RUL and Cycle condition vector 
        

        # real + fake data
        x = np.concatenate((real_samples, fake_samples))
        z1 = np.concatenate((fake_z1, fake_z1))

        # train discriminator1 to classify time series arrays 
        # as real/fake and recover latent code (z1)
        # joint training using discriminator part of advserial0 loss
        # and entropy1 loss
        metrics = dis1.train_on_batch(x, [y, z1])
        # log the overall loss only (use dis1.metrics_names)
        log = "%s [dis1_loss: %f]" % (log, metrics[0])
        dicta["dis1_loss"] = metrics[0]

        # adversarial training 
        # generate fake z0, labels
        fake_z0 = np.random.normal(scale=0.5, 
                                   size=[batch_size, z_dim])
        # input to generator0 is sampling fr real labels and
        # 50-dim z0 latent code
        gen0_inputs = [real_labels, fake_z0]

        # label fake feature0 as real (specifies whether real or not)
        
        y = np.ones([batch_size, 1])
    
        # train generator0 (thru adversarial) by fooling 
        # the discriminator
        # and approximating encoder1 feature0 generator
        # joint training: adversarial0, entropy0, conditional0
        metrics = adv0.train_on_batch(gen0_inputs,
                                      [y, fake_z0, real_labels[:,0:num_clusters]]) #Weil oben bei der erstellung des adv0 auch das array gesliced wurde
        fmt = "%s [adv0_loss: %f, enc1_acc: %f]"
        dicta["adv0_loss"] = metrics[0]
        dicta["enc1_acc"] = metrics[6]

        # log the overall loss and classification accuracy
        log = fmt % (log, metrics[0], metrics[6])

        # input to generator0 is real feature0 and 
        # 50-dim z0 latent code
        fake_z1 = np.random.normal(scale=0.5,
                                   size=[batch_size, z_dim])

        #[real_feature0,fake_z1, real_labels]
        gen1_inputs = [real_feature0,fake_z1, real_labels]#, con_vec]

        # train generator1 (thru adversarial) by fooling 
        # the discriminator and approximating encoder1 time series arrays 
        # source generator joint training: 
        # adversarial1, entropy1, conditional1
        metrics = adv1.train_on_batch(gen1_inputs,
                                      [y, fake_z1, real_feature0])
        # log the overall loss only
        log = "%s [adv1_loss: %f]" % (log, metrics[0])
        dicta["adv1_loss"] = metrics[0]

        losses.append(metrics)
        print('trainstep: ' + str(i))
        print(log)
        if (i + 1) % save_interval == 0:
            generators = (gen0, gen1)
            '''
            plot_ts(generators,
                        noise_params=noise_params,
                        show=False,
                        step=(i + 1),
                        model_name=model_name)
            '''
        #tv_plot.update({'dis0_loss': dicta["dis0_loss"], 'dis1_loss': dicta["dis1_loss"], 'adv0_loss': dicta["adv0_loss"], 'enc1_acc': dicta["enc1_acc"], 'adv1_loss': dicta["adv1_loss"]})
        #tv_plot.draw()
        
        
        #Acc Dis0 and Dis1
        
        acc_d0 = dis0.predict(fake_feature0)
        acc_d1 = dis1.predict(fake_samples)
        
        out_dis0 = acc_d0
        out_dis1 = acc_d1

        acc_d0 = np.sum(acc_d0[0])/ batch_size
        acc_d1 = np.sum(acc_d1[0])/ batch_size
        
        acc_d0 = 1-acc_d0
        acc_d1 = 1-acc_d1
        acc_fake.append([acc_d0,acc_d1])
        
        print('ACC dis0 Fake: ' + str(acc_d0) + ' and ACC dis1 Fake: ' + str(acc_d1)+ '\n')
        
        acc_d0 = dis0.predict(real_feature0)
        acc_d1 = dis1.predict(real_samples)
        
        acc_fake.append([acc_d0,acc_d1])
        
        acc_d0 = np.sum(acc_d0[0])/ batch_size
        acc_d1 = np.sum(acc_d1[0])/ batch_size
        
        acc_real.append([acc_d0,acc_d1])
        
        print('ACC dis0 Real: ' + str(1-acc_d0) + ' and ACC dis1 Real: ' + str(1-acc_d1)+ '\n')
    # save the modelis after training generator0 & 1
    # the trained generator can be reloaded for
    # future data generation
    gen0.save(model_name + "-gen1.h5")
    gen1.save(model_name + "-gen0.h5")
    #should log and plot acc disc0 and disc1
    acc_real = np.array(acc_real)
    #acc_real = np.flip(acc_real)
    #
    acc_fake = np.array(acc_fake, dtype=object)
    acc_fake = acc_fake[0::2,:]
    acc_fake = np.array(acc_fake, dtype=float)
    #acc_fake = np.flip(acc_fake)
    losses = np.array(losses)
    return  gen0, gen1, losses,acc_fake,acc_real

def plot_ts(generators,
                noise_params,
                show=False,
                step=0,
                model_name="gan"):
    """Generate fake time series arrays and plot them
    For visualization purposes, generate fake time series arrays
    then plot them in a square grid
    # Arguments
        generators (Models): gen0 and gen1 models for 
            fake time series arrays generation
        noise_params (list): noise parameters 
            (label, z0 and z1 codes)
        show (bool): Whether to show plot or not
        step (int): Appended to filename of the save time series arrays
        model_name (string): Model name
    """

    gen0, gen1 = generators
    noise_class, z0, z1 = noise_params
    feature0 = gen0.predict([noise_class, z0])
    tss = gen1.predict([feature0, z1])
    
    
def create_ts(generator0, generator1, condition):

    z0 = np.random.normal(scale=0.5, size=[condition.shape[0], Z_dim]) 
    z1 = np.random.normal(scale=0.5, size=[condition.shape[0], Z_dim])
    
    feature0 = generator0.predict([condition, z0])
    tss = generator1.predict([feature0,z1, condition])

    #condition = condition[:,features_n_c:condition.shape[1]]
    #condition = condition.reshape((condition.shape[0],seq_length,int(condition.shape[1]/seq_length)))
    #with col7
    #tss = np.concatenate([tss[:,:,0:2],condition[:,:,0:2],tss[:,:,4:7],condition[:,:,2:3],tss[:,:,8:19],condition[:,:,3:4]],axis=2) #concate generated Data with RUL and Cycle condition vector 
    #without col7
    #tss = np.concatenate([tss[:,:,0:2],condition[:,:,0:2],tss[:,:,4:7],tss[:,:,7:19],condition[:,:,2:3]],axis=2) #concate generated Data with RUL and Cycle condition vector 
    
    return tss
    
    
 #original model used 2000 (choose a low number if you)
gen0, gen1, losses ,acc_fake,acc_real  = build_and_train_models(train_steps=steps)


# Comment and uncomment to load/save models
gen1.save("gen0")
gen0.save_weights("gen1_weights")

#plot losses
fig, axes = plt.subplots(3, figsize=(12, 8))
fig.suptitle('Training Metrics')

axes[0].set_ylabel("Losses", fontsize=14)
axes[0].set_xlabel("Step", fontsize=14)
axes[0].plot(losses[:,0], color='red')
axes[0].plot(losses[:,1], color='blue')
axes[0].plot(losses[:,2], color='yellow')
axes[0].plot(losses[:,3], color='green')
axes[0].legend(['adv1_losses','gen1_losses','adv0_losses', 'adv1_losses']) #prüfen ob in Losses auch GAN losses sich befinden

axes[1].set_ylabel("Accuracy dis1", fontsize=14)
axes[1].set_xlabel("Epoch", fontsize=14)
axes[1].plot(acc_fake[:,1], color='red')
axes[1].plot(acc_real[:,1], color='blue')
axes[1].legend(['fake_acc','real_acc'])

axes[2].set_ylabel("Accuracy dis0", fontsize=14)
axes[2].set_xlabel("Epoch", fontsize=14)
axes[2].plot(acc_fake[:,0], color='red')
axes[2].plot(acc_real[:,0], color='blue')
axes[2].legend(['fake_acc','real_acc'])

plt.show()


tf.keras.utils.plot_model(
    gen1, to_file='model.png', show_shapes=False, show_layer_names=True,
    rankdir='TB', expand_nested=False, dpi=96
)
#Create fake Sequenz 

#sample_tss = create_ts(gen0, gen1, condition)
'''
if(True):
    train_data, train_label,max_value_rul,max_values = dp.preprocessing_cmapps(sl=seq_length, short=False)
    X = train_data
    clusters = cl.birch_clustering(np.concatenate([train_data,train_label],axis=2),features_n=features_n_c)
    clusters = np.reshape(clusters, [clusters.shape[0],1,1])
    clusters = np.repeat(clusters, seq_length, axis =1) #a cluster is calculated for one sequence, so it must be repeated for the sequence length, to fit the condition vector
    cycle = train_data[:,:,2:4] #get setting and cycle information of the real Dataset as part of condition vector
    
    col7 = train_data[:,:,7:8]
    Condtion_vector = np.concatenate([cycle,train_label,clusters], axis = 2)
    y = clusters[:,0,0] #get the first element out of a sequence, which is only needed for the label vector

    X = np.concatenate([X,train_label],axis=2)
    label = np.concatenate([cycle,col7,train_label],axis=2)#,col7
    del clusters
    #split in Train and Testdata, also split in data x and label y
    y = to_categorical(y)
    label = label.reshape([label.shape[0],label.shape[1]*label.shape[2]]) #flatten 2nd and 3rd dimension
    
    y = np.concatenate([y,label],axis= 1)
    tss, condition = create_ts(gen0,gen1,y)
    outfile = 'synthetic datasets/' +str(seq_length) + '_cmapps_fake_short_mtss2_1500steps_4_cond_lat50' #add Model Info to title
    np.save(outfile, tss)
'''  

if(True):
    n_datasets = 10
    all_data = np.concatenate([train_data,val_data,test_data],axis=0)
    all_label = np.concatenate([train_label,val_label,test_label],axis=0)

    
    #maybe add to categorical for cluster

    #create condition vector 
    #cluster formatieren von bsp. [3] zu [0,0,0,1,0,0,0,0,0,0]
    all_clusters = cl.birch_clustering(np.concatenate([all_data,all_label],axis=2))
    all_clusters = np.reshape(all_clusters, [all_clusters.shape[0],1,1])
    all_clusters = np.repeat(all_clusters, seq_length, axis =1)
    
    #clusters = cl.birch_clustering(np.concatenate([train_data,train_label],axis=2),features_n=features_n_c)
    y = to_categorical(all_clusters)
    y = y[0:train_label.shape[0],seq_length-1,:]
    y = np.concatenate([train_label[:,seq_length-1,:],y],axis= 1)

    syn_database = train_data
    s_label = train_label[:,seq_length-1,:]
    for i in range(n_datasets):
        
        t_database = create_ts(gen0,gen1,y)
        t_label = y[:,0:1]
        #t_database = t_database.numpy()
        syn_database = np.concatenate([syn_database,t_database],axis=0)
        s_label = np.concatenate([s_label,t_label],axis=0)
        
        
    syn_database = syn_database[:,:,0:25]
    outfile = 'synthetic datasets/' +str(seq_length) + 'Bearing_fake_data_10_mtss_cnn_lstm_ndis' #add Model Info to title
    np.save(outfile, syn_database)
    outfile = 'synthetic datasets/' +str(seq_length) + 'Bearing_fake_data_label_mtss_cnn_lstm_ndis' #add Model Info to title
    np.save(outfile, s_label)
