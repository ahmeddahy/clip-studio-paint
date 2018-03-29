import cv2
import numpy
import numpy as np
import pywt
import math
import scipy
from Sure_Shrink import Sure_Shrink
import matplotlib.pyplot as plt
from GaussianMixtureClassifier import GaussianModel
from GaussianMixtureClassifier import GaussianMixtureClassifier
import os
from CalcCrossCorrelation import calc_corr
from DWT_SURE_IDWT import dwt_sure_idwt
import bisect
#['bior1.1', 'bior1.3', 'bior1.5', 'bior2.2', 'bior2.4', 'bior2.6', 'bior2.8', 'bior3.1', 'bior3.3', 'bior3.5', 'bior3.7', 'bior3.9', 'bior4.4', 'bior5.5', 'bior6.8', 'cgau1', 'cgau2', 'cgau3', 'cgau4', 'cgau5', 'cgau6', 'cgau7', 'cgau8', 'cmor', 'coif1', 'coif2', 'coif3', 'coif4', 'coif5', 'coif6', 'coif7', 'coif8', 'coif9', 'coif10', 'coif11', 'coif12', 'coif13', 'coif14', 'coif15', 'coif16', 'coif17', 'db1', 'db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8', 'db9', 'db10', 'db11', 'db12', 'db13', 'db14', 'db15', 'db16', 'db17', 'db18', 'db19', 'db20', 'db21', 'db22', 'db23', 'db24', 'db25', 'db26', 'db27', 'db28', 'db29', 'db30', 'db31', 'db32', 'db33', 'db34', 'db35', 'db36', 'db37', 'db38', 'dmey', 'fbsp', 'gaus1', 'gaus2', 'gaus3', 'gaus4', 'gaus5', 'gaus6', 'gaus7', 'gaus8', 'haar', 'mexh', 'morl', 'rbio1.1', 'rbio1.3', 'rbio1.5', 'rbio2.2', 'rbio2.4', 'rbio2.6', 'rbio2.8', 'rbio3.1', 'rbio3.3', 'rbio3.5', 'rbio3.7', 'rbio3.9', 'rbio4.4', 'rbio5.5', 'rbio6.8', 'shan', 'sym2', 'sym3', 'sym4', 'sym5', 'sym6', 'sym7', 'sym8', 'sym9', 'sym10', 'sym11', 'sym12', 'sym13', 'sym14', 'sym15', 'sym16', 'sym17', 'sym18', 'sym19', 'sym20']
#print(pywt.wavelist())
path="Data set"
filename="04_original_enc20.avi"
video=os.path.join(path,filename)
cap = cv2.VideoCapture(video)
gray_frames = []
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret :
        gray = dwt_sure_idwt(frame)
        gray_frames.append(gray)
    else:
        break
'''orig=cv2.imread(os.path.join(path,'orig.jpg'))
orig_gray=cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
fake=cv2.imread(os.path.join(path,'fake.jpg'))
fake_gray=cv2.cvtColor(fake, cv2.COLOR_BGR2GRAY)'''
block=calc_corr(gray_frames[0],gray_frames[0])
size=gray_frames[0].shape
row_size=size[0]
col_size = size[1]
Blocks=[]
for i in range(0, row_size, 8):
    for j in range(0, col_size, 8):
        sub_matrix = block[i:i + 8, j:j + 8]
        list=[]
        for x in range(0,8):
            for y in range(0, 8):
                list.append(sub_matrix[x][y])
        Blocks.append(list)
x=GaussianModel()
Classes=GaussianMixtureClassifier(x,Blocks)
c=0
d=0
for i in Classes:
    if i==0:
        c+=1;
    else:
        d+=1;
print(c)
print(d)
print(Classes)
cap.release()
cv2.destroyAllWindows()