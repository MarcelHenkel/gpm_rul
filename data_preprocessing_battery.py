# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 10:17:07 2022

@author: POV GmbH
"""
from os import walk
import numpy as np
import pandas as pd
from scipy.io import loadmat
from os import listdir
import datetime
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler


import matplotlib.pyplot as plt


def preprocessing_battery(seq_length = 50):
    path = 'Data/battery/1st/'
    raw_data_paths = ['Data/battery/1st/', 'Data/battery/2nd/', 'Data/battery/3rd/', 'Data/battery/4th/', 'Data/battery/5th/', 'Data/battery/6th/']
    
    filenames_arr = []
    for r in raw_data_paths:
        fn = next(walk(r), (None, None, []))[2]  
        filenames_arr.append(fn)
    
    filenames = listdir(path)
    
    def load_data(battery, path, id_b):
      #mat = loadmat('battery_data/' + battery + '.mat')
      mat = loadmat(path+battery)
      battery = battery.replace('.mat','')
      #print('Total data in dataset: ', len(mat[battery][0, 0]['cycle'][0]))
      counter = 0
      dataset = []
      capacity_data = []
      
      for i in range(len(mat[battery][0, 0]['cycle'][0])):
        #if i %10 == 0:
        row = mat[battery][0, 0]['cycle'][0, i]
        if row['type'][0] == 'discharge':
          ambient_temperature = row['ambient_temperature'][0][0]
          year = int(row['time'][0][0])
          month = int(row['time'][0][1])
          day = int(row['time'][0][2])
          hour = int(row['time'][0][3])
          minute = int(row['time'][0][3])
          second = int(row['time'][0][4]) #+datetime.timedelta(seconds=int(row['time'][0][5]))
          '''
          date_time = datetime.datetime(int(row['time'][0][0]),
                                   int(row['time'][0][1]),
                                   int(row['time'][0][2]),
                                   int(row['time'][0][3]),
                                   int(row['time'][0][4])) + datetime.timedelta(seconds=int(row['time'][0][5]))
          '''
          data = row['data']
          capacity = data[0][0]['Capacity'][0][0]
          for j in range(len(data[0][0]['Voltage_measured'][0])):
            voltage_measured = data[0][0]['Voltage_measured'][0][j]
            current_measured = data[0][0]['Current_measured'][0][j]
            temperature_measured = data[0][0]['Temperature_measured'][0][j]
            current_load = data[0][0]['Current_load'][0][j]
            voltage_load = data[0][0]['Voltage_load'][0][j]
            time = data[0][0]['Time'][0][j]
            dataset.append([id_b, counter + 1, ambient_temperature,
                            voltage_measured, current_measured,
                            temperature_measured, current_load,
                            voltage_load, time, capacity])
          capacity_data.append([id_b, counter + 1, ambient_temperature, year, month, day, hour, minute, second, capacity])
          counter = counter + 1
      #print(dataset[0])
      return [pd.DataFrame(data=dataset,
                           columns=['id', 'cycle', 'ambient_temperature',
                                    'voltage_measured',
                                    'current_measured', 'temperature_measured',
                                    'current_load', 'voltage_load', 'time', 'capacity']),
              pd.DataFrame(data=capacity_data,
                           columns=['id','cycle', 'ambient_temperature', 'year', 'month', 'day', 'hour', 'minute', 'second',
                                    'capacity'])]
    
    train_data = []
    train_label = []
    val_data = []
    val_label = []
    test_data = []
    test_label = []
    
    
    i  = 0
    d = 0
    for fn in filenames_arr:
        for fname in fn:
            
            path = raw_data_paths[i]
            try:
                if(True and (i+d == 21 or i+d == 22 or i+d ==23 or i+d == 24 or i+d == 25 or i+d == 26 or i+d == 27)): #diese Messungen nicht verwenden
                    d = d+1
                else:
                    dataset, capacity = load_data(fname,path,i+d)
                    dataset = np.array(dataset, dtype=np.float64)
                    label = np.array(capacity, dtype=np.float64)
                    if(i+d == 1 or i+d==38 or i+d==16):
                        test_data.append(dataset)
                        test_label.append(dataset)
                    elif(i+d==2 or i+d==39 or i+d==17):
                        val_data.append(dataset)
                        val_label.append(dataset)
                    else:
                        train_data.append(dataset)
                        train_label.append(label)
                        
                    '''    
                    plot_df = capacity.loc[(capacity['cycle']>=1),['cycle','capacity']]
                    #sns.set_style("darkgrid")
        
                    
                    fig_acc = plt.figure(figsize=(12, 8))
                    plt.plot(plot_df['cycle'], plot_df['capacity'])
                    plt.plot([0.,len(capacity)], [1.4, 1.4])
                    plt.title('Capacity of Battery id: ' + str(i+d) + ' over Cycles')
                    plt.xlabel('cycle')
                    plt.ylabel('capacity [Ah]')
                    plt.legend(['Battery Capacity', 'Threshold Line'])
                    #plt.title('Discharge B0005')
                    plt.show()
                    '''
                    '''
                    plot_df = capacity.loc[(capacity['cycle']>=1),['cycle','capacity']]
                    fig_verify = plt.figure(figsize=(10, 5))
                    #plt.plot(pred_val_label_1*max_value_rul)#, color="orange")
                    #plt.plot(true_val_label_1*max_value_rul)#, color="blue")
                    #plt.plot(pred_val_label_2*max_value_rul)#, color="orange")
                    #plt.plot(true_val_label_2*max_value_rul)#, color="blue")
                    #plt.plot(pred_val_label_3*max_value_rul)#, color="orange")
                    #plt.plot(true_val_label_3*max_value_rul)#, color="blue")
                    plt.plot(plot_df['cycle'], plot_df['capacity'], color="green")

                    #plt.title('Predicted and real remaining capaicty')
                    plt.title('predicted and real remaining capacity (test set)')
                    plt.ylabel('remaining capacity [Ah]')
                    plt.xlabel('cylce')
                    #plt.legend(['pred bat1', 'real bat1','pred bat2', 'real bat2','pred bat3', 'real bat3'], loc='upper right')
                    #plt.legend([ 'prediction battery B0006','real battery B0006','prediction battery B0034', 'real battery B0034','prediction battery B0056','real battery B0056'], loc='lower right')
                    plt.legend([ 'battery B00' +str(i+d)])
                    plt.show()
                    '''
                    d = d+1
            except:
                pass
    
        i = i+1
    
    def generate_sequence(data):
        train = []
        for element in data:
            for i in range(1,int(np.max(element[:,1]))):
                
                np_seq = np.where(element[:,1] == i) #get rows with same cycle
                np_seq = np.array(np_seq[0])
        
                start = int(np_seq[0]) #row where i == id and time min
                end = int(np_seq[-1]) #row where i == id and time max
                
                indices = np.array(np.linspace(start,end,num=seq_length), dtype=int)
                seq =[]
                
                for ii in indices:
                    seq.append(np.reshape(element[ii,:],[1,element.shape[1]]))
                
                seq = np.array(seq, dtype = np.float64)    
                seq = np.reshape(seq, [seq.shape[0],seq.shape[2]])
                
                train.append(seq)
        train = np.array(train, dtype= np.float64)
        return train
            
    #### generate sequences in seq_length and create label vector
    
    train_data = generate_sequence(train_data)
    val_data = generate_sequence(val_data)
    test_data = generate_sequence(test_data)
    

    #### min max scaler
    min_max_scaler = MinMaxScaler()
    
    train_rows = int(train_data.shape[0])
    val_rows = int(val_data.shape[0])
    test_rows = int(test_data.shape[0])
    
    all_data = np.concatenate([train_data,val_data,test_data], axis=0)
    max_rul_value = np.max(all_data[:,:,-1])
    x = all_data.shape[0]
    y = all_data.shape[1]
    z = all_data.shape[2]
    all_data = np.reshape(all_data,[all_data.shape[0]*all_data.shape[1],all_data.shape[2]])
    all_data = min_max_scaler.fit_transform(all_data) 
    all_data = np.reshape(all_data,[x,y,z])
    train_data = all_data[0:train_rows,:,:]
    val_data = all_data[train_rows:train_rows+val_rows,:,:]
    test_data =all_data[train_rows+val_rows:train_rows+val_rows+test_rows,:,:]
    #get rul (capacity) label vectors 
    train_label = train_data[:,0,-1]
    train_label = np.reshape(train_label, [train_label.shape[0],1])
    train_data = train_data[:,:,0:-1]#drop rul column
    val_label = val_data[:,0,-1]
    val_label = np.reshape(val_label, [val_label.shape[0],1])
    val_data = val_data[:,:,0:-1]#drop rul column
    test_label = test_data[:,0,-1]
    test_label = np.reshape(test_label, [test_label.shape[0],1])
    test_data = test_data[:,:,0:-1]#drop rul column

    return train_data, train_label, val_data, val_label, test_data, test_label, max_rul_value
#Auflösung der Datensätze verringern auf ein zentel
#zeit in eine Zahl umwandeln bzw. Zeit stoppen in stunden

#capacity enthält in der letzten Spalte (cap[:,9]) die RUL Werte
#Spalte 0: Id
#Spalte 1: Cycle
#Spalte 2: ambient_temperature
#Spalte 3-8: Datum
#Spalte 9: Capacity => RUL 

#data spalten: 
#Spalte 1: Id
#Spalte 2: Cycle
#Spalte 3: ambient_temperature
#Spalte : voltage_measured
#Spalte : current_measured
#Spalte : temperature_measured
#Spalte : current_load
#Spalte : voltage_load
#Spalte : Time absolute bzw. verstrichene Zeit
#Spalte : Capacity

#Problem: eine Sequenz = Cycle ist unterschiedlich lang
#bsp Messreihe 0 aus train_data hat 168 Zyklen = Sequenzen
#aber ein Zyklus ist 196 und manchmal 192 Reihen lang und haben jeweil nur eine Kapazität also RUL Wert zugewiesen
#in anderen Messreihen ist ein Zyklus teilweise 500 reihen lang

#Vorschlag: setze Sequenz auf 50 reihen länge fest. Dann nehme die gesamtlänge einer Sequenz und teile diese durch 50. 
#Der Abstand, der sich dann ergibt, in diesem wird dann abgetastet
#Die Angabe der verstrichenen Zeit beinhaltet dann immer noch die Information über die tatsächliche länge der Sequenz.