# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 10:45:21 2022

@author: Marcel Henkel

Based on the paper: time-series regeneration with convolutional recurrent generative adversarial network for remaining useful life estimation" 
(https://www.semanticscholar.org/paper/Time-Series-Regeneration-With-Convolutional-Network-Zhang-Qin/55dd7715123a5ada27ad2b8db996a54170dcf9bd)
"""



from tensorflow import keras
from tensorflow.keras import layers
from keras.utils import to_categorical
#from tensorflow_docs.vis import embed
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import random
from tqdm import tqdm
from IPython import display
import data_preprocessing_cmapps as cdp
import data_preprocessing_bearing as bdp
import data_preprocessing_battery as badp
import clustering as cl

#label discriminator 1 = real, 0 = fake
epochen = 2000
min_epochen = 1500
#################################################################################
disc_steps = 3 #repeat discriminator x times training for each train step
gen_steps = 1 #repeat generator x times training for each train step

lr_d = 0.00005 #learningrate Discriminator
lr_g = 0.00005 #learningrate Generator
#ori lr 0.00001
combined_acc = 1.05
latent_dim = 20 #size of latent vector = noise vector
num_channels = 1 #number of channel of discriminator refering to  size of label / conditional Label 
np.random.seed(1234)  
PYTHONHASHSEED = 0
batch_size = 32
num_classes = 1 #hier ist es statt 10 => 1, da die RUL Information ein nomierter Skalar ist #10 #Anzahl der unterschiedlichen Conditions
eag = False #enable eagerly-mode / Debugging-Mode
seq_length = sequence_length = 24 #Anzahl der sukezessive aufeinanderfolgende Messungen, welche den Input des Discriminators darstellen
discriminator_in_channels = 1  #dimension of discriminator input layer
repeat_rul_label_gen = 10
generator_in_channels = latent_dim +repeat_rul_label_gen #dimension of generator input layer
gen_output_abs = False
fix_latent_vektor = False #add label vector
load_working_weights = False
training = True

#norm_value = 2173
noise_dim = [batch_size, sequence_length, latent_dim]
use_break_point = True
seed = tf.random.normal(noise_dim)
gen_load_path = 'saved weights all/CNN LSTM 100 Ep'
# read training data - it is the aircraft engine run-to-failure data.


dataset_n = 'FD001'
flat_rul = True
max_res = 20

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

    #train_label = np.expand_dims(train_label[:,sequence_length-1],axis=1)
    #val_label = np.expand_dims(val_label[:,sequence_length-1],axis=1)
    #test_label = np.expand_dims(test_label[:,sequence_length-1],axis=1)
    
    train_label = (train_label/m_rul)
    val_label = (val_label/m_rul)
    test_label = (test_label/m_rul)
    
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

#train_data, train_label, val_data,val_label, test_data, test_label, max_value_rul = bearing(sequence_length, False, 120, flat = True)
train_data, train_label, val_data,val_label, test_data, test_label, max_value_rul = battery(sequence_length)
#train_data, train_label, val_data,val_label, test_data, test_label, max_value_rul = cmapss(sequence_length,dataset_n)
'''
all_data = np.concatenate([train_data,val_data,test_data],axis=0)
all_label = np.concatenate([train_label,val_label,test_label],axis=0)

#maybe add to categorical for cluster

#create condition vector 
#cluster formatieren von bsp. [3] zu [0,0,0,1,0,0,0,0,0,0]
all_clusters = cl.birch_clustering(np.concatenate([all_data,all_label],axis=2))
all_clusters = np.reshape(all_clusters, [all_clusters.shape[0],1,1])
all_clusters = np.repeat(all_clusters, sequence_length, axis =1) #a cluster is calculated for one sequence, so it must be repeated for the sequence length, to fit the condition vector
#cycle = train_data[:,:,3:4] #get third column of seq array, while maintaining shape
#train_data = np.delete(train_data,3,2) #delete cycle column
#condtion_vector = np.concatenate([cycle,train_label,clusters], axis = 2)

all_clusters = to_categorical(all_clusters)
'''
if(True):
    slicing = np.r_[0:200]#,2000:2100,3000:3100]#,4000:4100,5000:5100,6000:6100,7000:7100,8000:8100,9000:9100,10000:10100,11000:11100,12000:12100,13000:13100,14000:14100,15000:15100]#,16000:16100,17000:17100,18000:18100,19000:19100]
    #slicing = np.r_[0:100]#,2000:2100,3000:3100]#,4000:4100,5000:5100,6000:6100,7000:7100,8000:8100,9000:9100,10000:10100,11000:11100,12000:12100,13000:13100,14000:14100,15000:15100]#,16000:16100,17000:17100,18000:18100,19000:19100]
    #slicing = np.r_[0:100,1000:1100]#,2000:2100,3000:3100]#,4000:4100,5000:5100,6000:6100,7000:7100,8000:8100,9000:9100,10000:10100,11000:11100,12000:12100,13000:13100,14000:14100,15000:15100]#,16000:16100,17000:17100,18000:18100,19000:19100]
    #slicing = np.r_[0:100,1000:1100,2000:2100,3000:3100]#,4000:4100,5000:5100,6000:6100,7000:7100,8000:8100,9000:9100,10000:10100,11000:11100,12000:12100,13000:13100,14000:14100,15000:15100]#,16000:16100,17000:17100,18000:18100,19000:19100]
    #slicing = np.r_[0:100,1000:1100,2000:2100,3000:3100,4000:4100,5000:5100,6000:6100,7000:7100,8000:8100,9000:9100,10000:10100,11000:11100,12000:12100,13000:13100,14000:14100,15000:15100]#,16000:16100,17000:17100,18000:18100,19000:19100]
    train_data = train_data[slicing,:,:]
    train_label = train_label[slicing,:]
'''
indices_label = train_label<0.85# and train_label>0.05#np.where(train_label<0.85)


indices_label = indices_label[:,seq_length-1]
#condtion_vector = np.concatenate([train_label,all_clusters[0:train_label.shape[0],:]], axis = 2) #only the cluster information for the train_label
condtion_vector = np.expand_dims(train_label[indices_label,:],axis=2) #only the cluster information for the train_label

train_data = train_data[indices_label,:,:]


'''
condtion_vector = train_label


condtion_vector = np.expand_dims(condtion_vector,axis=2)
condtion_vector = np.repeat(condtion_vector,seq_length,axis=1)
'''
add_vector = np.concatenate([train_data[:,:,0:1],train_data[:,:,2:3],train_data[:,:,6:7],train_data[:,:,7:8]],axis=2)

condtion_vector = np.concatenate([condtion_vector,add_vector],axis=2)

train_data = np.concatenate([train_data[:,:,1:2],train_data[:,:,3:6],train_data[:,:,8:9]],axis=2)
'''
save_train_val_test_data = False

ts_size = train_data.shape[2] #length of timeseries / number of sensors
label_dim = condtion_vector.shape[2] #Größe eines RUL Label vektors (1= Skalar, 2= 2 Skalare pro RUL information)


if(save_train_val_test_data):

    
    outfile = 'Data/' + 'bearing_train_data_24_2' 
    np.save(outfile, train_data)
    
    outfile = 'Data/' + 'bearing_train_label_24_2' 
    np.save(outfile, train_label)
    
    outfile = 'Data/' + 'bearing_val_data_24_2' 
    np.save(outfile, val_data)
    
    outfile = 'Data/' + 'bearing_val_label_24_2' 
    np.save(outfile, val_label)
    
    outfile = 'Data/' + 'bearing_test_data_24_2' 
    np.save(outfile, test_data)
    
    outfile = 'Data/' + 'bearing_test_label_24_2' 
    np.save(outfile, test_label)
    

dataset = tf.data.Dataset.from_tensor_slices((train_data, condtion_vector))
dataset = dataset.shuffle(buffer_size=1024).batch(batch_size) #ist der shuffle entlang der ersten Achse? hoffentlich, bitte prüfen



print(f"Shape of training labels: {dataset}")

#reading and transforming data - done


#create the discriminator. 

def make_discriminator_model():
    model = tf.keras.Sequential(name="discriminator",)
    model.add(layers.InputLayer((sequence_length,ts_size+label_dim)))
    #model.add(layers.Flatten())
    #model.add(layers.Conv1D(filters=18, kernel_size=2, strides=1, padding='same'))
    #model.add(layers.Conv1D(filters=18, kernel_size=2, strides=1, padding='same'))
    #model.add(layers.Conv1D(filters=18, kernel_size=2, strides=1, padding='same'))
    model.add((layers.LSTM(units= int(100), return_sequences=(True)))) 
    model.add(layers.Dropout(0.2))
    model.add(layers.LSTM(units= 75, dropout=0.2, return_sequences=(True))) 
    model.add((layers.LSTM(units= 50, dropout=0.1, return_sequences=(False))))
    
    model.add(layers.Dense(20))
    model.add(layers.Dense(1))#,activation='softmax'))


    return model


# Create the generator.
#Ebenso Generator anpassen
def make_generator_model():
    model = tf.keras.Sequential(name="generator",)
    model.add(layers.InputLayer((sequence_length,latent_dim+label_dim))) 
    
    model.add(layers.Conv1D(filters=ts_size, kernel_size=2, strides=1, padding='same'))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2,strides=1, padding='same'))
    #model.add(layers.ReLu())
    #model.add(layers.MaxPooling1D(pool_size=(2), strides=(2), padding='same'))
    model.add(layers.Conv1D(filters=ts_size, kernel_size=2, strides=1, padding='same'))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2,strides=1, padding='same'))
    #model.add(layers.ReLu())
    #model.add(layers.MaxPooling1D(pool_size=(2), strides=(2), padding='same'))
    model.add(layers.Conv1D(filters=ts_size, kernel_size=2, strides=1, padding='same'))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2,strides=1, padding='same'))
    #model.add(layers.ReLu())
    #model.add(layers.MaxPooling1D(pool_size=(2), strides=(2), padding='same'))
    #model.add(layers.GRU(units= 200, return_sequences=(True)))
    
    model.add((layers.LSTM(units= 200, return_sequences=(True))))
    model.add(layers.Dropout(0.1))
    model.add((layers.LSTM(units= 150, return_sequences=(True))))
    model.add(layers.Dropout(0.1))
    model.add((layers.LSTM(units= 100, return_sequences=(True))))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(units=ts_size*2))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(units=ts_size*1.5))
    model.add(layers.LeakyReLU(alpha=0.1))
    #model.add(layers.Dropout(0.25))
    model.add(layers.Dense(units=ts_size,activation='relu'))
    #model.add(layers.Reshape((sequence_length,ts_size)))
    
    #model.add(layers.LSTM(units= ts_size, return_sequences=(True)))
    
    return model

generator     = make_generator_model()
discriminator = make_discriminator_model()
#generator.run_eagerly = eag
#discriminator.run_eagerly = eag
    
generator.summary()
discriminator.summary()

#test_show(generator, discriminator)

#################################################################################
#          Prepare metrics for logging
#################################################################################

# !rm -rf ./logs/

### discriminator loss ###
disc_log_dir = 'logs/gradient_tape/disc_loss'
disc_summary_writer = tf.summary.create_file_writer(disc_log_dir)
disc_losses = tf.keras.metrics.Mean('disc_loss', dtype=tf.float32)
disc_losses_list = []

### discriminator accuracy ###
fake_disc_accuracy = tf.keras.metrics.BinaryAccuracy('fake_disc_accuracy')
real_disc_accuracy = tf.keras.metrics.BinaryAccuracy('real_disc_accuracy')
fake_disc_accuracy_list, real_disc_accuracy_list = [], []

### generator loss ###
gen_log_dir = 'logs/gradient_tape/gen_loss'
gen_summary_writer = tf.summary.create_file_writer(gen_log_dir)
gen_losses = tf.keras.metrics.Mean('gen_loss', dtype=tf.float32)
gen_losses_list = []


#################################################################################
#          Prepare loss functions and optimizers
#################################################################################

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_data, fake_data):
    generated_and_real_ts_rul_label = tf.concat([real_data,fake_data], axis =0)
    labels =tf.concat([tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0)
        
    indices = tf.range(start=0, limit=tf.shape(generated_and_real_ts_rul_label)[0], dtype=tf.int32)
    shuffled_indices = tf.random.shuffle(indices)
    
    generated_and_real_ts_rul_label = tf.gather(generated_and_real_ts_rul_label, shuffled_indices)
    labels = tf.gather(labels, shuffled_indices)
    
    predictions = discriminator(generated_and_real_ts_rul_label)
    
    total_loss = cross_entropy(labels, predictions)
    
    return total_loss

def generator_loss(real_data, fake_data):
    generated_and_real_ts_rul_label = tf.concat([real_data, fake_data], axis =0)
    labels =tf.concat([tf.zeros((batch_size, 1)),tf.ones((batch_size, 1))], axis=0)
        
    indices = tf.range(start=0, limit=tf.shape(generated_and_real_ts_rul_label)[0], dtype=tf.int32)
    shuffled_indices = tf.random.shuffle(indices)
    
    generated_and_real_ts_rul_label = tf.gather(generated_and_real_ts_rul_label, shuffled_indices)
    labels = tf.gather(labels, shuffled_indices)
    
    predictions = discriminator(generated_and_real_ts_rul_label)
    gen_loss = cross_entropy(labels, predictions)

    return gen_loss 

generator_optimizer = tf.keras.optimizers.Adam(lr_g)
discriminator_optimizer = tf.keras.optimizers.Adam(lr_d)


# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(real_ts, dim, condition_vector):
    noise = tf.random.normal(dim)
    condition_vector = tf.cast(condition_vector, dtype=tf.float32)
    real_ts = tf.cast(real_ts, dtype=tf.float32)
    #print('rul_labels shape:' + str(rul_labels.shape))
    noise_and_condition_vector = tf.concat([noise,condition_vector], axis=2)
    real_ts_and_condition_vector = tf.concat([real_ts,condition_vector], axis=2)

    for i in range(disc_steps):
        with tf.GradientTape() as disc_tape:

            generated_ts = generator(noise_and_condition_vector, training=True)
            generated_ts_and_condition_vector = tf.concat([generated_ts,condition_vector], axis=2)
            #real_output = discriminator(real_ts_and_condition_vector, training=True)
            #fake_output = discriminator(generated_ts_and_condition_vector, training=True)

            disc_loss = discriminator_loss(real_ts_and_condition_vector, generated_ts_and_condition_vector)

        
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
     
        
    for i in range(gen_steps):
        with tf.GradientTape() as gen_tape:
    
            generated_ts = generator(noise_and_condition_vector, training=True)
            generated_ts_and_condition_vector = tf.concat([generated_ts,condition_vector], axis=2)
            
            
            
            gen_loss = generator_loss(real_ts_and_condition_vector, generated_ts_and_condition_vector)
        
            gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
            generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

        
    ### for tensorboard ###
    disc_losses.update_state(disc_loss)
    real_output = discriminator(real_ts_and_condition_vector, training=True) #müssten diese nichts False sein?
    fake_output = discriminator(generated_ts_and_condition_vector, training=True)
    fake_disc_accuracy.update_state(tf.zeros_like(fake_output), fake_output)
    real_disc_accuracy.update_state(tf.ones_like(real_output), real_output)

    ### for tensorboard ###
    gen_losses.update_state(gen_loss)
    #######################
    
    


def train(dataset, epochs, dim):
    
    for epoch in tqdm(range(epochs)):
        
    
        for batch in dataset:
            real_ts, labels = batch
            if((labels.shape[0]%batch_size) == 0):

                train_step(real_ts, dim, labels)
            
        disc_losses_list.append(disc_losses.result().numpy())
        gen_losses_list.append(gen_losses.result().numpy())
        
        fake_disc_accuracy_list.append(fake_disc_accuracy.result().numpy())
        real_disc_accuracy_list.append(real_disc_accuracy.result().numpy())
        print('\n')
        print('disc_acc_fake_data: ' + str(fake_disc_accuracy.result().numpy())+'\n')
        print('disc_acc_real_data: ' + str(real_disc_accuracy.result().numpy())+'\n')
        print('gen_loss: ' +str(gen_losses.result().numpy())+'\n')
        print('disc_loss: ' +str(disc_losses.result().numpy())+'\n')
        if(epoch>min_epochen and use_break_point and ((fake_disc_accuracy.result().numpy()<0.95 and real_disc_accuracy.result().numpy()<0.22) or ((fake_disc_accuracy.result().numpy()+real_disc_accuracy.result().numpy())<combined_acc))): #Abbruchbedingung für Overfitting
            #Bedingungen durch Variabeln ersetzen
            epochen = epoch
            break
        ### for tensorboard ###
        with disc_summary_writer.as_default():
            tf.summary.scalar('loss', disc_losses.result(), step=epoch)
            tf.summary.scalar('fake_accuracy', fake_disc_accuracy.result(), step=epoch)
            tf.summary.scalar('real_accuracy', real_disc_accuracy.result(), step=epoch)
            
        with gen_summary_writer.as_default():
            tf.summary.scalar('loss', gen_losses.result(), step=epoch)
            
        disc_losses.reset_states()        
        gen_losses.reset_states()
        
        fake_disc_accuracy.reset_states()
        real_disc_accuracy.reset_states()
        #######################

        # Save the model every 5 epochs
#         if (epoch + 1) % 5 == 0:
#             generate_and_save_ecg(generator, epochs, seed, False)
#             checkpoint.save(file_prefix = checkpoint_prefix)

    # Generate after the final epoch
    display.clear_output(wait=True)
    
    #generate_and_save_ecg(generator, epochs, seed, False, labels)
    

generator_optimizer = tf.keras.optimizers.Adam(lr_g)
discriminator_optimizer = tf.keras.optimizers.Adam(lr_d)
if(training):
    #train(dataset, epochen, noise_dim)  
    train(dataset, epochen, noise_dim)
    save_path_weigths = 'drop_out_high_weigths_epo_' + str(epochen) +'lrg_'+str(lr_g) + 'lrd_'+ str(lr_d) +'_latent_dim_' +str(latent_dim)+'batch_sz_'+str(batch_size)+'seq_len_'+str(sequence_length)

    #generator.save_weights('gen'+save_path_weigths)
    #discriminator.save_weights('dis'+save_path_weigths)
    generator.save('gen'+save_path_weigths)
    discriminator.save('dis'+save_path_weigths)
else:
    generator.load_weights(gen_load_path)

# %reload_ext tensorboard
# %tensorboard --logdir logs/gradient_tape

fig, axes = plt.subplots(2, figsize=(12, 8))
fig.suptitle('Training Metrics')

axes[0].set_ylabel("Losses", fontsize=14)
axes[0].set_xlabel("Epoch", fontsize=14)
axes[0].plot(disc_losses_list, color='red')
axes[0].plot(gen_losses_list, color='blue')
axes[0].legend(['disc_losses','gen_losses'])

axes[1].set_ylabel("Accuracy", fontsize=14)
axes[1].set_xlabel("Epoch", fontsize=14)
axes[1].plot(fake_disc_accuracy_list, color='red')
axes[1].plot(real_disc_accuracy_list, color='blue')
axes[1].legend(['fake_acc','real_acc'])
plt.show()


def generate_ts(num_seq, condition):
    
    condition = tf.cast([condition], dtype=tf.float32)
    condition = tf.reshape([condition],(1,sequence_length,label_dim))
    dim = [num_seq, sequence_length, latent_dim]
    noise = tf.random.normal(dim)
    noise_and_condition_vector = tf.concat([noise,condition], axis=2)
    ts = generator(noise_and_condition_vector)
    ts_and_label = tf.concat([ts,condition], axis=2)
    
    return ts_and_label

#sampled_ts_late7  = generate_ts(1,condtion_vector[100,:,:]).numpy()

#train_data, train_label,max_value_rul,max_values = dp.preprocessing_cmapps(sl=sequence_length, short=False)

train_data, train_label, val_data,val_label, test_data, test_label, max_value_rul = battery(sequence_length)



condtion_vector_v = train_label
condtion_vector_v = np.expand_dims(condtion_vector_v,axis=2)
condtion_vector_v = np.repeat(condtion_vector_v,seq_length,axis=1)
#add_vector = np.concatenate([train_data[:,:,0:1],train_data[:,:,2:3],train_data[:,:,6:7],train_data[:,:,7:8]],axis=2)
#condtion_vector_v = np.concatenate([condtion_vector_v,add_vector],axis=2)
'''
condtion_vector_t = test_label
condtion_vector_t = np.expand_dims(condtion_vector_t,axis=2)
condtion_vector_t = np.repeat(condtion_vector_t,seq_length,axis=1)
add_vector = np.concatenate([test_label[:,:,0:1],test_label[:,:,2:3],test_label[:,:,6:7],test_label[:,:,7:8]],axis=2)
condtion_vector_t = np.concatenate([condtion_vector_t,add_vector],axis=2)


condition_vector_vt = np.concatenate([condtion_vector_v,condtion_vector_t],axis=0)
'''

def generate_dataset(rul_label):
    dataset = np.empty([1, sequence_length,ts_size])
    for i in range(round(rul_label.shape[0])):
        rul_label_t = rul_label[i,:,:]
        rul_label_t = np.reshape(rul_label_t, [1,sequence_length,label_dim]) #add one dimension in the first place [30,1] => [1,30,1]
        dim = [1, sequence_length, latent_dim]
        noise = tf.random.normal(dim)
        
        noise_and_condition_vector = tf.concat([noise,rul_label_t], axis=2)
        
        ts = generator(noise_and_condition_vector)
        #ts_and_label = tf.concat([ts,rul_label_t], axis=2)
        dataset = np.concatenate((dataset,ts.numpy()), axis=0)
        
    
    return  dataset[1:dataset.shape[0],:,:] #drop first element from initialization
if(True):
    n_datasets = 1
    syn_database = train_data
    s_label = train_label
    

    for i in range(n_datasets):
        t_database = generate_dataset(condtion_vector_v)
        t_label = train_label #rul col
        #t_database = np.concatenate([add_vector[:,:,0:1],t_database[:,:,0:1],add_vector[:,:,1:2],t_database[:,:,1:4],add_vector[:,:,2:4],t_database[:,:,4:6]],axis=2)

    outfile = 'synthetic datasets/' +'battery_200rows_cr_gan_data_ep1000_2' #add Model Info to title
    np.save(outfile, t_database)
    outfile = 'synthetic datasets/' + 'battery_200rows_cr_gan_label_ep1000_2' #add Model Info to title
    np.save(outfile, t_label)

if(False):
    #train_data, train_label,max_value_rul,max_values = dp.preprocessing_cmapps(sl=sequence_length, short=True)
    #create condition vector 
    #cluster formatieren von bsp. [3] zu [0,0,0,1,0,0,0,0,0,0]
    clusters = cl.birch_clustering(np.concatenate([train_data,train_label],axis=2))
    clusters = np.reshape(clusters, [clusters.shape[0],1,1])
    clusters = np.repeat(clusters, sequence_length, axis =1) #a cluster is calculated for one sequence, so it must be repeated for the sequence length, to fit the condition vector
    cycle = train_data[:,:,3:4] #get third column of seq array, while maintaining shape
    train_data = np.delete(train_data,3,2) #delete cycle column
    condtion_vector = np.concatenate([cycle,train_label,clusters], axis = 2)
    real_dataset = np.concatenate([train_data,condtion_vector],axis=2)
    outfile = 'synthetic datasets/' +str(sequence_length) + '_cmapps_real_ultra_short' #add Model Info to title
    np.save(outfile, real_dataset)

#plot heatmap of bearings through fourier transformation via 4 subplots
if(False):
    data_sum_rows_train_1 = train_data[0:599,:,:].sum(axis=1)
    data_sum_rows_train_2 = train_data[600:1199,:,:].sum(axis=1)
    data_sum_rows_val = val_data.sum(axis=1)
    data_sum_rows_test = test_data.sum(axis=1)
    
    #delete background noise, which is column 3
    data_sum_rows_train_1 = np.delete(data_sum_rows_train_1,3,axis=1)
    data_sum_rows_train_2 = np.delete(data_sum_rows_train_2,3,axis=1)
    data_sum_rows_val = np.delete(data_sum_rows_val,3,axis=1)
    data_sum_rows_test = np.delete(data_sum_rows_test,3,axis=1)

    x_axis_scale = np.linspace(0,100,num=599)
    y_axis_scale = np.linspace(0,20000,num=25*20)

    fig, axs = plt.subplots(2,2, figsize=(7, 7))
    #fig = plt.figure(figsize=(10, 10))
    #plt.title('Fourier trainsformation bearing 4 dataset (test)')
    
    #axs[0, 0].figure(figsize=(7, 7))
    axs[0, 0].imshow(np.rot90((np.log(np.repeat(data_sum_rows_train_1,20,axis=1)))), cmap='viridis', aspect="auto", extent=[0,100,0,20000])
    #axs[0, 0].set_xlim([0, 100])
    #axs[0, 0].set_ylim([0, 20000])
    axs[0, 0].set_title('bearing 1 (training)')

    #axs[0, 1].figure(figsize=(7, 7))
    axs[0, 1].imshow(np.rot90((np.log(np.repeat(data_sum_rows_train_2,20,axis=1)))), cmap='viridis', aspect="auto", extent=[0,100,0,20000])
    #axs[0, 1].set_xlim([0, 100])
    #axs[0, 1].set_ylim([0, 20000])
    axs[0, 1].set_title('bearing 2 (training)')
    
    #axs[1, 0].figure(figsize=(7, 7))
    axs[1, 0].imshow(np.rot90((np.log(np.repeat(data_sum_rows_val,20,axis=1)))), cmap='viridis', aspect="auto", extent=[0,100,0,20000])
    #axs[1, 0].set_xlim([0, 100])
    #axs[1, 0].set_ylim([0, 20000])
    axs[1, 0].set_title('bearing 3 (validation)')
    
    #axs[1, 1].figure(figsize=(7, 7))
    axs[1, 1].imshow(np.rot90((np.log(np.repeat(data_sum_rows_test,20,axis=1)))), cmap='viridis', aspect="auto", extent=[0,100,0,20000])
    #axs[1, 1].set_xlim([0, 100])
    #axs[1, 1].set_ylim([0, 20000])
    axs[1, 1].set_title('bearing 1 (test)')
    #plt.colorbar()
    
    #plt.ylabel('frequencies')
    #plt.xlabel('runtime')
    
    for ax in axs.flat:
        ax.set(xlabel='runtime [h]', ylabel='frequency [Hz]')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()
    plt.show()
    
