import cv2
import numpy
import numpy as np
import pywt
import math
import scipy
import matplotlib.pyplot as plt
import os
import bisect
def visu_shrink(coefficients):
    numofcoefficients=len(coefficients)
    threshold=np.var(coefficients)*math.sqrt(2 * math.log2(numofcoefficients))
    #coefficients = pywt.threshold(coefficients, threshold, 'hard')
    coefficients = pywt.threshold(coefficients, threshold, 'soft')
    return