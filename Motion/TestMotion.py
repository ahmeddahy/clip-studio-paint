import cv2
import numpy as np
from typing import List
from sklearn import svm
import os
from Motion import *
clf = svm.SVC(kernel='poly', max_iter=1000, gamma=3.4, degree=8)
m=Motion(clf)
path="Data set"
filename="04_original_enc30.avi"
video=os.path.join(path,filename)
clf = svm.SVC(kernel='poly', max_iter=1000, gamma=3.4, degree=8)
m=Motion(clf)
m.read_video(video)
m.compute_features()
print("Da7y Finished ...")