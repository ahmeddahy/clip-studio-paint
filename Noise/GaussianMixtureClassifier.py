import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

def ProbabilityDensityFunction(x,mean,variance):
    '''
    :param x: sample value
    :param mean: mean of class
    :param variance: variance of the class
    :opertioan: calcluate the probability of belonging x sample to this class by use PDF function
    :return: probability value of belonging the sample to this class
    '''
    pdf=1
    for i in range(0,len(x),1):
     pdf=pdf*(1/math.sqrt(2*math.pi*variance[i]))*math.exp((-1*math.pow(x[i]-mean[i],2))/(2*variance[i]))
    return pdf

def GaussianModel():
    '''
    opertion:construct the model and set the number of class
    :return: object of Gaussina mixture model
    '''
    gmm = GaussianMixture(n_components=2)
    return gmm

def GaussianMixtureClassifier(gmm,Data):
    '''
    :param gmm: object of Gaussina mixture model
    :param Data: dataset samples
    :return: 2D array each row represent one cluster data
    '''
    NumberOFSamples=Data.__len__()
    gmm.fit(Data)# fit the date in the model to split it to 2 classes
    Clutsers=[[] for i in range(gmm.n_components)]
    for sample in range(0,NumberOFSamples,1):
     Max=0
     idx=0
     for cluster in range(0,gmm.n_components,1):
        variance=gmm.covariances_[cluster].diagonal() #extract the variance of each class from its covariance matrix
        # calculate the probability of belonging the sample to this class
        pro=ProbabilityDensityFunction(Data[sample],gmm.means_[cluster],variance)
        if(pro>Max):#assign the sample to the class that has the higher probability
           Max=pro
           idx=cluster
     Clutsers[idx].append(Data[sample])
    return Clutsers



