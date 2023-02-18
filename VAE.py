# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 12:45:39 2022

@author: Marcel Henkel
"""

import keras
from keras import layers
import matplotlib.pyplot as plt
import numpy as np
import data_preprocessing as dp



from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, concatenate, InputLayer,UpSampling1D, Reshape
from keras.models import Sequential
from keras.optimizers import Adam

class Dense_VAE:
  def __init__(self,n_features,seq_length, batch_size, intermediate_dim=None,epochen=100,latent_dim=30, learning_rate= 0.0004,loss='mse'):
    self.n_features = n_features
    self.epochen = epochen
    self.batch_size = batch_size
    self.seq_length = seq_length
    self.optimizer = Adam(learning_rate=learning_rate)
    self.loss = loss
    self.n_features = n_features
    self.original_dim = (seq_length,n_features)
    self.original_dim_n = seq_length*n_features
    if(intermediate_dim==None):
        self.intermediate_dim = (seq_length*n_features/4)
    else:
        self.intermediate_dim = intermediate_dim
    self.latent_dim = latent_dim
  '''  
  def s_loss(self, decoder, encoder):
    inputs = keras.Input(shape=(original_dim,))  
    outputs = decoder(encoder(inputs)[2])  
    
      
    reconstruction_loss = keras.losses.binary_crossentropy(inputs, outputs)
    reconstruction_loss *= original_dim
    kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)  
    
    return vae_loss
  '''   
  def build_encoder(self):
    
    n_features = self.n_features
    encoder = Sequential(name='encoder')
    
    # Encoder
    encoder.add(InputLayer((self.seq_length, self.n_features)))
    encoder.add(Reshape((self.seq_length*self.n_features,)))
    encoder.add(Dense(units=self.intermediate_dim))
    encoder.add(Dense(units=self.latent_dim))
    '''
    #encoder.add(UpSampling1D(size=repeat_n))
    encoder.add(LSTM(n_features, activation='relu', return_sequences=True)) # input_shape=(timesteps, n_features*repeat_n)
    encoder.add(LSTM(intermediate_dim, activation='relu', return_sequences=True))
    #encoder.add(LSTM(latent_dim, activation='relu', return_sequences=True))
    #or return_sequence=False and RepeatVector(seq_length)
    encoder.add(LSTM(latent_dim, activation='relu', return_sequences=False))
    encoder.add(RepeatVector(seq_length))
    
    encoder.compile(optimizer=self.optimizer, loss=self.loss)
    '''
    encoder.summary()
    self.encoder = encoder
    
  def build_decoder(self):
    n_features = self.n_features
    decoder = Sequential(name='decoder')
    '''
    # Decoder
    decoder.add(LSTM(latent_dim, activation='relu',input_shape=(seq_length, latent_dim), return_sequences=True))
    decoder.add(LSTM(intermediate_dim, activation='relu', return_sequences=True))
    decoder.add(LSTM(n_features, activation='relu', return_sequences=True))
    decoder.add(TimeDistributed(Dense(n_features,activation='sigmoid')))
    decoder.compile(optimizer=self.optimizer, loss=self.loss)
    '''
    decoder.add(InputLayer((self.latent_dim,)))
    decoder.add(Dense(units=self.latent_dim))
    decoder.add(Dense(units=self.intermediate_dim))
    decoder.add(Dense(units=self.seq_length*self.n_features))
    decoder.add(Reshape((self.seq_length,self.n_features)))
    decoder.summary()
    self.decoder = decoder
    
    
    
  def fit(self, X, validation_data, lr=0.0004, batch_size=32):
    self.timesteps = X.shape[1]
    self.build_encoder()
    self.build_decoder()
    self.vae = Sequential(name='vae')
    self.vae.add(self.encoder) 
    self.vae.add(self.decoder)
    #self.s_loss(self.encoder, self.decoder)
    self.vae.compile(optimizer=self.optimizer, loss=self.loss)
    
    self.vae.fit(X, X, epochs=self.epochen, batch_size=batch_size, validation_data=validation_data)
    self.vae.summary()
    
  def predict(self, X):
    # = np.expand_dims(X, axis=2)
    output_X = self.model.predict(X)
    reconstruction = np.squeeze(output_X)
    return np.linalg.norm(X - reconstruction, axis=-1)

  def encode(self, X):
    return self.encoder.predict(X)

  def decode(self, X):
    return self.decoder.predict(X)
  
  def plot(self, scores, timeseries, threshold=0.95):
    sorted_scores = sorted(scores)
    threshold_score = sorted_scores[round(len(scores) * threshold)]
    
    plt.title("Reconstruction Error")
    plt.plot(scores)
    plt.plot([threshold_score]*len(scores), c='r')
    plt.show()
    
    anomalous = np.where(scores > threshold_score)
    normal = np.where(scores <= threshold_score)
    
    plt.title("Anomalies")
    plt.scatter(normal, timeseries[normal][:,-1], s=3)
    plt.scatter(anomalous, timeseries[anomalous][:,-1], s=5, c='r')
    plt.show()
    
class LSTM_VAE:
  def __init__(self,n_features,seq_length, batch_size,epochen=100,latent_dim=30, learning_rate= 0.0004,loss='mse'):
    self.n_features = n_features
    self.epochen = epochen
    self.batch_size = batch_size
    self.seq_length = seq_length
    self.optimizer = Adam(learning_rate=learning_rate)
    self.loss = loss
    self.n_features = n_features
    self.original_dim = (seq_length,n_features)
    self.original_dim_n = seq_length*n_features
    self.intermediate_dim = int(seq_length*n_features/4)
    self.latent_dim = latent_dim

  def build_encoder(self):
    
    n_features = self.n_features
    encoder = Sequential(name='encoder')
    
    # Encoder
    encoder.add(InputLayer((self.seq_length, self.n_features)))
    encoder.add(Reshape((self.seq_length*self.n_features,)))


    #encoder.add(UpSampling1D(size=repeat_n))
    encoder.add(LSTM(n_features, activation='relu', return_sequences=True)) # input_shape=(timesteps, n_features*repeat_n)
    encoder.add(LSTM(intermediate_dim, activation='relu', return_sequences=True))
    #encoder.add(LSTM(latent_dim, activation='relu', return_sequences=True))
    #or return_sequence=False and RepeatVector(seq_length)
    encoder.add(LSTM(latent_dim, activation='relu', return_sequences=False))
    encoder.add(RepeatVector(seq_length))
    
    encoder.compile(optimizer=self.optimizer, loss=self.loss)

    encoder.summary()
    self.encoder = encoder
    
  def build_decoder(self):
    n_features = self.n_features
    decoder = Sequential(name='decoder')
    '''
    # Decoder
    decoder.add(LSTM(latent_dim, activation='relu',input_shape=(seq_length, latent_dim), return_sequences=True))
    decoder.add(LSTM(intermediate_dim, activation='relu', return_sequences=True))
    decoder.add(LSTM(n_features, activation='relu', return_sequences=True))
    decoder.add(TimeDistributed(Dense(n_features,activation='sigmoid')))
    decoder.compile(optimizer=self.optimizer, loss=self.loss)
    '''
    decoder.add(InputLayer((self.latent_dim,)))
    decoder.add(Dense(units=self.latent_dim))
    decoder.add(Dense(units=self.intermediate_dim))
    decoder.add(Dense(units=self.seq_length*self.n_features))
    decoder.add(Reshape((self.seq_length,self.n_features)))
    decoder.summary()
    self.decoder = decoder
    
    
    
  def fit(self, X, validation_data, lr=0.0004, batch_size=32):
    self.timesteps = X.shape[1]
    self.build_encoder()
    self.build_decoder()
    self.vae = Sequential(name='vae')
    self.vae.add(self.encoder) 
    self.vae.add(self.decoder)
    #self.s_loss(self.encoder, self.decoder)
    self.vae.compile(optimizer=self.optimizer, loss=self.loss)
    
    self.vae.fit(X, X, epochs=self.epochen, batch_size=batch_size, validation_data=validation_data)
    self.vae.summary()
    
  def predict(self, X):
    # = np.expand_dims(X, axis=2)
    output_X = self.model.predict(X)
    reconstruction = np.squeeze(output_X)
    return np.linalg.norm(X - reconstruction, axis=-1)

  def encode(self, X):
    return self.encoder.predict(X)

  def decode(self, X):
    return self.decoder.predict(X)
  
  def plot(self, scores, timeseries, threshold=0.95):
    sorted_scores = sorted(scores)
    threshold_score = sorted_scores[round(len(scores) * threshold)]
    
    plt.title("Reconstruction Error")
    plt.plot(scores)
    plt.plot([threshold_score]*len(scores), c='r')
    plt.show()
    
    anomalous = np.where(scores > threshold_score)
    normal = np.where(scores <= threshold_score)
    
    plt.title("Anomalies")
    plt.scatter(normal, timeseries[normal][:,-1], s=3)
    plt.scatter(anomalous, timeseries[anomalous][:,-1], s=5, c='r')
    plt.show()
    
class CNN_VAE:
  def __init__(self,n_features,seq_length, batch_size,epochen=100,latent_dim=30, learning_rate= 0.0004,loss='mse'):
    self.n_features = n_features
    self.epochen = epochen
    self.batch_size = batch_size
    self.seq_length = seq_length
    self.optimizer = Adam(learning_rate=learning_rate)
    self.loss = loss
    self.n_features = n_features
    self.original_dim = (seq_length,n_features)
    self.original_dim_n = seq_length*n_features
    self.intermediate_dim = int(seq_length*n_features/4)
    self.latent_dim = latent_dim
  '''  
  def s_loss(self, decoder, encoder):
    inputs = keras.Input(shape=(original_dim,))  
    outputs = decoder(encoder(inputs)[2])  
    
      
    reconstruction_loss = keras.losses.binary_crossentropy(inputs, outputs)
    reconstruction_loss *= original_dim
    kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)  
    
    return vae_loss
  '''   
  def build_encoder(self):
    
    n_features = self.n_features
    encoder = Sequential(name='encoder')
    
    # Encoder
    encoder.add(InputLayer((self.seq_length, self.n_features)))
    encoder.add(Reshape((self.seq_length*self.n_features,)))
    encoder.add(Dense(units=self.intermediate_dim))
    encoder.add(Dense(units=self.latent_dim))
    '''
    #encoder.add(UpSampling1D(size=repeat_n))
    encoder.add(LSTM(n_features, activation='relu', return_sequences=True)) # input_shape=(timesteps, n_features*repeat_n)
    encoder.add(LSTM(intermediate_dim, activation='relu', return_sequences=True))
    #encoder.add(LSTM(latent_dim, activation='relu', return_sequences=True))
    #or return_sequence=False and RepeatVector(seq_length)
    encoder.add(LSTM(latent_dim, activation='relu', return_sequences=False))
    encoder.add(RepeatVector(seq_length))
    
    encoder.compile(optimizer=self.optimizer, loss=self.loss)
    '''
    encoder.summary()
    self.encoder = encoder
    
  def build_decoder(self):
    n_features = self.n_features
    decoder = Sequential(name='decoder')
    '''
    # Decoder
    decoder.add(LSTM(latent_dim, activation='relu',input_shape=(seq_length, latent_dim), return_sequences=True))
    decoder.add(LSTM(intermediate_dim, activation='relu', return_sequences=True))
    decoder.add(LSTM(n_features, activation='relu', return_sequences=True))
    decoder.add(TimeDistributed(Dense(n_features,activation='sigmoid')))
    decoder.compile(optimizer=self.optimizer, loss=self.loss)
    '''
    decoder.add(InputLayer((self.latent_dim,)))
    decoder.add(Dense(units=self.latent_dim))
    decoder.add(Dense(units=self.intermediate_dim))
    decoder.add(Dense(units=self.seq_length*self.n_features))
    decoder.add(Reshape((self.seq_length,self.n_features)))
    decoder.summary()
    self.decoder = decoder
    
    
    
  def fit(self, X, validation_data, lr=0.0004, batch_size=32):
    self.timesteps = X.shape[1]
    self.build_encoder()
    self.build_decoder()
    self.vae = Sequential(name='vae')
    self.vae.add(self.encoder) 
    self.vae.add(self.decoder)
    #self.s_loss(self.encoder, self.decoder)
    self.vae.compile(optimizer=self.optimizer, loss=self.loss)
    
    self.vae.fit(X, X, epochs=self.epochen, batch_size=batch_size, validation_data=validation_data)
    self.vae.summary()
    
  def predict(self, X):
    # = np.expand_dims(X, axis=2)
    output_X = self.model.predict(X)
    reconstruction = np.squeeze(output_X)
    return np.linalg.norm(X - reconstruction, axis=-1)

  def encode(self, X):
    return self.encoder.predict(X)

  def decode(self, X):
    return self.decoder.predict(X)
  
  def plot(self, scores, timeseries, threshold=0.95):
    sorted_scores = sorted(scores)
    threshold_score = sorted_scores[round(len(scores) * threshold)]
    
    plt.title("Reconstruction Error")
    plt.plot(scores)
    plt.plot([threshold_score]*len(scores), c='r')
    plt.show()
    
    anomalous = np.where(scores > threshold_score)
    normal = np.where(scores <= threshold_score)
    
    plt.title("Anomalies")
    plt.scatter(normal, timeseries[normal][:,-1], s=3)
    plt.scatter(anomalous, timeseries[anomalous][:,-1], s=5, c='r')
    plt.show()    
    
'''
seq_length = 24 
seq_array, label_array,max_value_rul,max_values = dp.preprocessing_cmapps(sl=seq_length, short=True)
x_train = seq_array#np.concatenate([seq_array,label_array], axis= 2)
#split in validation data und training data
split = int(x_train.shape[0]*0.9)
x_test = x_train[split:-1,:,:]
x_train = x_train[0:split,:,:]
    

lstm_autoencoder = LSTM_Autoencoder(n_features=19,seq_length=seq_length,batch_size=32, loss='mse', )
lstm_autoencoder.fit(x_train, validation_data=(x_test, x_test))
#scores = lstm_autoencoder.predict(x_test)
#lstm_autoencoder.plot(scores, x_test, threshold=0.95)

x_test_reconstructed = lstm_autoencoder.decode(lstm_autoencoder.encode(x_test))

'''
