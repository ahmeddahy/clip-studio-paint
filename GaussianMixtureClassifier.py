import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
def ProbabilityDensityFunction(x,mean,variance):
    pdf=1
    for i in range(0,len(x),1):
     pdf=pdf*(1/math.sqrt(2*math.pi*variance[i]))*math.exp((-1*math.pow(x[i]-mean[i],2))/(2*variance[i]))
    return pdf
def GaussianModel():
    gmm = GaussianMixture(n_components=2)
    return gmm
def GaussianMixtureClassifier(gmm,Data):
    NumberOFSamples=Data.__len__()
    gmm.fit(Data)
    DataClutserNumber=[]
    for sample in range(0,NumberOFSamples,1):
     Max=0
     idx=0
     for cluster in range(0,2,1):
        variance=gmm.covariances_[cluster].diagonal()
        pro=ProbabilityDensityFunction(Data[sample],gmm.means_[cluster],variance)
        if(pro>Max):
           Max=pro
           idx=cluster
     DataClutserNumber.append(idx)
    return DataClutserNumber



