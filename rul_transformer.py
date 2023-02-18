# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 09:25:03 2022
@author: Marcel Henkel
"""
import numpy as np
import data_preprocessing_cmapps as cdp
import keras.backend as K
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import tensorflow as tf
import data_preprocessing_cmapps as cdp
import data_preprocessing_bearing as bdp
import data_preprocessing_battery as badp
from sklearn.metrics import mean_squared_error

def readucr(filename):
    data = np.loadtxt(filename, delimiter="\t")
    y = data[:, 0]
    x = data[:, 1:]
    return x, y.astype(int)

seq_length = sequence_length = 24
epochs = 200
batch_size = 128
mlp_units = 256
head_size = 256
lr = 1e-4
dropout_enc = 0.25
kernel_width = 3 #filters, might be height
num_heads = 6

mlp_dropout = 0.3 #dropout rate dense layer 
n_classes = 1
early_stop_patience = 50
flat_rul = True


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

#train_data, train_label, val_data,val_label, test_data, test_label, max_value_rul = bearing(sequence_length, False, 120, flat = True)
#train_data, train_label, val_data,val_label, test_data, test_label, max_value_rul = battery(sequence_length)
train_data, train_label, val_data,val_label, test_data, test_label, max_value_rul = cmapss(sequence_length,dataset_n)

    
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
    outfile = 'synthetic datasets/' + '24cmapss_FD001_mtss_data_col_del.npy' #add Model Info to title
    s_data = np.load(outfile)
    outfile = 'synthetic datasets/' + '24cmapss_FD001_mtss_label_col_del.npy' #add Model Info to title
    s_label = np.load(outfile)
    #s_label = s_label[:,seq_length-1,:]
    #concate with original trainingdata
    #train_data = np.concatenate([train_data,s_data],axis=0)
"""
## Build the model
Our model processes a tensor of shape `(batch size, sequence length, features)`,
where `sequence length` is the number of time steps and `features` is each input
timeseries.
You can replace your classification RNN layers with this one: the
inputs are fully compatible!
"""


"""
We include residual connections, layer normalization, and dropout.
The resulting layer can be stacked multiple times.
The projection layers are implemented through `keras.layers.Conv1D`.
"""


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Attention and Normalization
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res

def r2_keras(y_true, y_pred):
   
    res =  K.sum(K.square( y_true - y_pred ))
    tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - res/(tot + K.epsilon()) )


"""
The main part of our model is now complete. We can stack multiple of those
`transformer_encoder` blocks and we can also proceed to add the final
Multi-Layer Perceptron classification head. Apart from a stack of `Dense`
layers, we need to reduce the output tensor of the `TransformerEncoder` part of
our model down to a vector of features for each data point in the current
batch. A common way to achieve this is to use a pooling layer. For
this example, a `GlobalAveragePooling1D` layer is sufficient.
"""


def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(n_classes, activation="relu")(x)
    return keras.Model(inputs, outputs)


"""
## Train and evaluate
"""

input_shape = train_data.shape[1:]

model = build_model(
    input_shape,
    head_size=head_size, #encoder size
    num_heads=num_heads, #number of MultiHeadAttention in encoder
    ff_dim=kernel_width, #filter/Kernel size used in encoder 
    num_transformer_blocks=5, #number of encoder
    mlp_units=[mlp_units],
    mlp_dropout=mlp_dropout, #Dense Layer Dropoutrate
    dropout=dropout_enc, #Encoder Dropout
)

model.compile(
    loss='mse',
    optimizer=keras.optimizers.Adam(learning_rate=lr),
    metrics=['mse',r2_keras],
)
model.summary()

def scheduler(epoch,lr):
    if epoch < 4:
        return lr
    else:
        return lr * tf.math.exp(-0.03)


my_callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss",verbose=1, restore_best_weights=True, patience=early_stop_patience),
        #tf.keras.callbacks.LearningRateScheduler(scheduler)
]
history = model.fit(
    train_data,
    train_label,
    validation_split=0.1,
    epochs=10,
    batch_size=batch_size,
    validation_data=(val_data,val_label),
    verbose=2,
)
history = model.fit(
    train_data,
    train_label,
    validation_split=0.1,
    epochs=epochs,
    batch_size=batch_size,
    callbacks=my_callbacks,
    validation_data=(val_data,val_label),
    verbose=2,
)

model.evaluate(val_data, val_label, verbose=1)

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
#MSE losses
fig_acc = plt.figure(figsize=(6, 6))
#plt.plot(history.history['mean_absolute_error'])
#plt.plot(history.history['val_mean_absolute_error'])
#plt.plot(save_train_loss)
plt.plot(history.history['mse'])
plt.plot(history.history['val_mse'])
#plt.plot(save_val_loss)

plt.title('Transformer RUL model losses')
plt.ylabel('MSE')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')

#plt.legend(['train real', 'validation real', 'train crgan data', 'validation crgan data'], loc='upper right')
plt.show()
fig_acc.savefig("model_mse.png")


"""
## Conclusions
In about 110-120 epochs (25s each on Colab), the model reaches a training
accuracy of ~0.95, validation accuracy of ~84 and a testing
accuracy of ~85, without hyperparameter tuning. And that is for a model
with less than 100k parameters. Of course, parameter count and accuracy could be
improved by a hyperparameter search and a more sophisticated learning rate
schedule, or a different optimizer.
You can use the trained model hosted on [Hugging Face Hub](https://huggingface.co/keras-io/timeseries_transformer_classification)
and try the demo on [Hugging Face Spaces](https://huggingface.co/spaces/keras-io/timeseries_transformer_classification).
"""

def score_cal(y_hat, val_label):
    d   = y_hat - val_label
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
    return np.sum((y_pred-y_true))/y_pred.shape[0]



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
    
    '''
   
    battery_val_row_1 = 166
    battery_val_row_2 = 205
    
    true_val_label_1 = test_label[0:battery_val_row_1,:]
    pred_val_label_1 = y_pred_test[0:battery_val_row_1,:]
    
    true_val_label_2 = test_label[battery_val_row_1:battery_val_row_2,:]
    pred_val_label_2 = y_pred_test[battery_val_row_1:battery_val_row_2,:]
    
    true_val_label_3 = test_label[battery_val_row_2:-1,:]
    pred_val_label_3= y_pred_test[battery_val_row_2:-1,:]
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
    plt.legend([ 'prediction battery B0006','real battery B0006','prediction battery B0034', 'real battery B0034','prediction battery B0056','real battery B0056'], loc='lower right')
    #plt.legend([ 'battery B0006'])
    plt.show()
    
    '''
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
    d = y_pred_val*max_value_rul - val_label*max_value_rul
    plt.hist(d, bins='auto')  
    plt.title('Error distribution - test set FD001')
    plt.ylabel('f')
    plt.xlabel("Error: $RUL_{hat}$ - RUL [cycle]")
    plt.show()
    '''
