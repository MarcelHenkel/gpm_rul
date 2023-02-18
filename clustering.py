# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 10:48:30 2022

@author: Marcel Henkel
"""


# synthetic classification dataset



from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.cluster import Birch

#Overview of clustering methods: https://scikit-learn.org/stable/modules/clustering.html
#wie können hier ganze sequenzen berücksichtigt werden? => Sequenz als eine Zeile schreiben?
#Eine Achse als RUL also diese verketten mit den Spalten der Ergebnisse der Cluster Algorithmen? => So wird es auf jeden Fall auch bei der Erstellung des Conditional Vektors gemacht
#Wie viel feature soll der Conditional Vektor besitzen?




def agglomerative_clustering(data,features_n=10,sequence=True):
    if(sequence): #Flatten Data to 2D
        data = data.reshape(data.shape[0],data.shape[1]*data.shape[2])
    model = AgglomerativeClustering(n_clusters=features_n)
    # fit model and predict clusters
    yhat = model.fit_predict(data)
   
    return yhat

def birch_clustering(data,features_n=10,sequence=True):
    if(sequence): #Flatten Data to 2D
        data = data.reshape(data.shape[0],data.shape[1]*data.shape[2])
    #Birch
    model = Birch(threshold=0.01, n_clusters=features_n)
    # fit the model
    model.fit(data)
    # assign a cluster to each example
    yhat = model.predict(data)

    return yhat

def gaussian_mixture_model(data,features_n=10,sequence=True):
    # define the model
    model = GaussianMixture(n_components=features_n)
    # fit the model
    model.fit(data)
    # assign a cluster to each example
    yhat = model.predict(data)
    
    return yhat
        
        
    
    
'''
import data_preprocessing as dp
import numpy as np
from numpy import where
from matplotlib import pyplot
from sklearn.datasets import make_classification
from numpy import unique
from sklearn.cluster import KMeans
from sklearn.cluster import OPTICS
from sklearn.cluster import SpectralClustering

sequence_length = 20

seq_array, label_array,max_value_rul,max_values = dp.preprocessing_cmapps(sl=sequence_length)
if(True): #slicing in sequences every 1000 element for 100 elements
    slicing = np.r_[0:100,1000:1100,2000:2100,3000:3100,4000:4100,5000:5100,6000:6100,7000:7100,8000:8100,9000:9100,10000:10100,11000:11100,12000:12100,13000:13100,14000:14100,15000:15100,16000:16100,17000:17100,18000:18100]
    test_data_set = seq_array[slicing,:,:]
    test_label =label_array[slicing,:,:]
    test_data_set = np.concatenate([test_data_set,test_label], axis = 2)
    test_data_set = test_data_set.reshape([test_data_set.shape[0],(sequence_length)*test_data_set.shape[2]])
elif(False): #Take whole Dataset, but only first Element in a Sequnce
    test_data_set = seq_array[:,0,0:19]
    test_label =label_array[:,0,0:1]
    if(True): #True label will be concatenated
        test_data_set = np.concatenate([test_data_set,test_label], axis = 1)
else:#Take whole Dataset
    if(True): #True label will be concatenated
        test_data_set = np.concatenate([seq_array,label_array], axis = 2)
    test_data_set = test_data_set.reshape([test_data_set.shape[0],(sequence_length)*test_data_set.shape[2]])
    
##### Agglomerative Clustering (Hierarchical Clustering)

# define the model
model = AgglomerativeClustering(n_clusters=10)
# fit model and predict clusters
yhat = model.fit_predict(test_data_set)
# retrieve unique clusters
clusters = unique(yhat)
# create scatter plot for samples from each cluster
for cluster in clusters:
	# get row indexes for samples with this cluster
	row_ix = where(yhat == cluster)
	# create scatter of these samples
	pyplot.scatter(test_data_set[row_ix, 3]*max_value_rul, test_data_set[row_ix, 19]*max_value_rul)
pyplot.title('Agglomerative Clustering')
pyplot.legend(['Cluster a','Cluster b','Cluster c','Cluster d','Cluster e'])
pyplot.xlabel('Total Runtime')
pyplot.ylabel('RUL')
# show the plot
pyplot.show()


#K Means
model = KMeans(n_clusters=10)
# fit the model
model.fit(test_data_set)
# assign a cluster to each example
yhat = model.predict(test_data_set)
# retrieve unique clusters
clusters = unique(yhat)
# create scatter plot for samples from each cluster
for cluster in clusters:
	# get row indexes for samples with this cluster
	row_ix = where(yhat == cluster)
	# create scatter of these samples
	pyplot.scatter(test_data_set[row_ix, 3]*max_value_rul, test_data_set[row_ix, 19]*max_value_rul)
pyplot.title('K Means Clustering')
pyplot.legend(['Cluster a','Cluster b','Cluster c','Cluster d','Cluster e'])
pyplot.xlabel('Total Runtime')
pyplot.ylabel('RUL')
# show the plot
pyplot.show()

#OPTICS

# define the model
model = OPTICS(eps=0.8, min_samples=10)
# fit model and predict clusters
yhat = model.fit_predict(test_data_set)
# retrieve unique clusters
clusters = unique(yhat)
# create scatter plot for samples from each cluster
for cluster in clusters:
	# get row indexes for samples with this cluster
	row_ix = where(yhat == cluster)
	# create scatter of these samples
	pyplot.scatter(test_data_set[row_ix, 3]*max_value_rul, test_data_set[row_ix, 19]*max_value_rul)
pyplot.title('OPTICS Clustering')
pyplot.legend(['Cluster a','Cluster b','Cluster c','Cluster d','Cluster e'])
pyplot.xlabel('Total Runtime')
pyplot.ylabel('RUL')
# show the plot
pyplot.show()

#Spectral Clustering

# define the model
model = SpectralClustering(n_clusters=5)
# fit model and predict clusters
yhat = model.fit_predict(test_data_set)
# retrieve unique clusters
clusters = unique(yhat)
# create scatter plot for samples from each cluster
for cluster in clusters:
	# get row indexes for samples with this cluster
	row_ix = where(yhat == cluster)
	# create scatter of these samples
	pyplot.scatter(test_data_set[row_ix, 3]*max_value_rul, test_data_set[row_ix, 19]*max_value_rul)
pyplot.title('Spectral Clustering')
pyplot.legend(['Cluster a','Cluster b','Cluster c','Cluster d','Cluster e'])
pyplot.xlabel('Total Runtime')
pyplot.ylabel('RUL')
# show the plot
pyplot.show()

#Gaussian Mixture Model

# define the model
model = GaussianMixture(n_components=10)
# fit the model
model.fit(test_data_set)
# assign a cluster to each example
yhat = model.predict(test_data_set)
# retrieve unique clusters
clusters = unique(yhat)
# create scatter plot for samples from each cluster
for cluster in clusters:
	# get row indexes for samples with this cluster
	row_ix = where(yhat == cluster)
	# create scatter of these samples
	pyplot.scatter(test_data_set[row_ix, 3]*max_value_rul, test_data_set[row_ix, 19]*max_value_rul)
pyplot.title('Gaussian Mixture Model Clustering')
pyplot.legend(['Cluster a','Cluster b','Cluster c','Cluster d','Cluster e'])
pyplot.xlabel('Total Runtime')
pyplot.ylabel('RUL')
# show the plot
pyplot.show()

#Birch
model = Birch(threshold=0.01, n_clusters=10)
# fit the model
model.fit(test_data_set)
# assign a cluster to each example
yhat = model.predict(test_data_set)
# retrieve unique clusters
clusters = unique(yhat)
# create scatter plot for samples from each cluster
for cluster in clusters:
	# get row indexes for samples with this cluster
	row_ix = where(yhat == cluster)
	# create scatter of these samples
	pyplot.scatter(test_data_set[row_ix, 3]*max_value_rul, test_data_set[row_ix, 19]*max_value_rul)
pyplot.title('Birch Clustering')
pyplot.legend(['Cluster a','Cluster b','Cluster c','Cluster d','Cluster e'])
pyplot.xlabel('Total Runtime')
pyplot.ylabel('RUL')
# show the plot
pyplot.show()
'''
