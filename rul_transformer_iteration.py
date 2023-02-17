
import numpy as np
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


dataset_lists = ['bearing']

seq_length = sequence_length = 24
epochs = 200
batch_size = 32
mlp_units = [256]
dropout_enc = 0.25
kernel_width = 4 #filters, might be height
num_heads = 6
head_size = 256
mlp_dropout = 0.4 #dropout rate dense layer 
n_classes = 1
lr_rul = 1e-4
#layer list als liste anderer Parameter (siehe Overleaf)
#mlp units, head_size,kernel_width,num_heads, num_transformer_blocks,dropout_enc,mlp_ dropout
layer_list = [[[256],256,3,5,5,0.25,0.3],[[256],256,3,5,5,0.25,0.3]]
scheduler_start = 20
early_stop_patience = 40
flat_rul = True



def battery(seq_length):
    return badp.preprocessing_battery(seq_length)

def cmapss(seq_length, dataset_n):

    
    train_data, train_label, val_data,val_label, test_data, test_label = cdp.preprocessing_cmapps(dataset_n=dataset_n, seq_length = seq_length)
    
    
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
    

    #ad dummy dimension
    train_label = np.reshape(train_label,[train_label.shape[0],1])
    val_label = np.reshape(val_label,[val_label.shape[0],1])
    test_label = np.reshape(test_label,[test_label.shape[0],1])
    
    return train_data, train_label, val_data,val_label, test_data, test_label, max_value_rul

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

#gleich die Zeile ausführen und danach auskommentieren

#train_data, train_label, val_data,val_label, test_data, test_label, max_value_rul = bearing(sequence_length, False, flat = True)
train_data, train_label, val_data,val_label, test_data, test_label, max_value_rul = battery(sequence_length)
#train_data, train_label, val_data,val_label, test_data, test_label, max_value_rul = cmapss(sequence_length,dataset_n)

    




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



def scheduler(epoch,lr):
    if epoch < 4:
        return lr
    else:
        return lr * tf.math.exp(-0.03)




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

ITERATIONS = 10
#mlp units, head_size,num_heads,kernel_width, num_transformer_blocks,dropout_enc,mlp_ dropout
results = pd.DataFrame(columns=['MSE val', 'std_MSE val', '`MSE train', 'std_MSE train',   # bigger std means less robust
                                'epochs',  'batch_size', 
                                'mlp units', 'head_size','num_heads,kernel_width', 'num_transformer_blocks','dropout_enc','mlp_ dropout',
                                'sequence_length', 'dataset','patience','flat rul', 'lr'])  

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
        layer = layer_list[i]
        
        
        #train_data, train_label, val_data,val_label, test_data, test_label, max_value_rul = cmapss(sequence_length,datan)
        
        
                                     
        #input_shape = (sequence_length, train_data.shape[2])
        input_shape = (1,train_data.shape[1])
        
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
        
            input_shape = train_data.shape[1:]

            model = build_model(
                input_shape,
                head_size=layer_list[i][1], #encoder size
                num_heads=layer_list[i][3], #number of MultiHeadAttention in encoder
                ff_dim=layer_list[i][2], #filter/Kernel size used in encoder 
                num_transformer_blocks=layer_list[i][4], #number of encoder
                mlp_units=[layer_list[i][0]],
                mlp_dropout=layer_list[i][6], #Dense Layer Dropoutrate
                dropout=layer_list[i][5], #Encoder Dropout
            )

            model.compile(
                loss='mse',
                optimizer=keras.optimizers.Adam(learning_rate=lr_rul),
                metrics=['mse',r2_keras],
            )

            history = model.fit(
                train_data,
                train_label,
                epochs=5,
                batch_size=batch_size,
                validation_data=(val_data,val_label),
                verbose=2,
            )
            history = model.fit(
                train_data,
                train_label,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=my_callbacks,
                validation_data=(val_data,val_label),
                verbose=2,
            )
            # fit the network

            
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
                 'epochs':np.mean(early_stop),
                 'mlp units':str(layer_list[i][0]), 
                 'head_size':str(layer_list[i][1]),
                 'num_heads':str(layer_list[i][3]),
                 'kernel_width':str(layer_list[i][2]), 
                 'num_transformer_blocks':str(layer_list[i][4]),
                 'dropout_enc':str(layer_list[i][5]),
                 'mlp_ dropout':str(layer_list[i][6]),
                 'batch_size':batch_size, 'sequence_length':sequence_length, 'dataset':datan, 'patience':early_stop_patience,
                 'flat rul':str(flat_rul), 'lr': str(lr_rul)
            }
        
            results = results.append(pd.DataFrame(d, index=[0]), ignore_index=True)
        
            results.to_csv('transformer_rul_hyperparameter_bearing5.csv', sep=';')
        except:
            #dann waren es nur flat line predictions
            pass
    
