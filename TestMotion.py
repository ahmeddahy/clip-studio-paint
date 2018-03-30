import cv2
import numpy as np
from typing import List
from sklearn import svm

from Motion import *
clf = svm.SVC(kernel='poly', max_iter=1000, gamma=3.4, degree=8)
m=Motion(clf)
frame=m.read_video('01_forged.avi')
resultDa7y=m.markov_features(frame)
print("Da7y Finished ...")
print(resultDa7y)
