import cv2
import numpy
import numpy as np
import pywt
import math
def calc_corr(noise_matrix1,noise_matrix2):
    noise_matrix1 = np.float32(noise_matrix1)
    noise_matrix2 = np.float32(noise_matrix2)
    size=noise_matrix1.shape
    tmp=[]
    row_size = size[0]
    col_size = size[1]
    sub_matrix1=numpy.zeros((8, 8))
    mask_matrix = numpy.zeros((8, 8))
    for i in range (0,row_size,8):
        tmp1=[]
        for j in range(0, col_size, 8):
            sub_matrix1=noise_matrix1[i:i+8,j:j+8]
            mask_matrix = noise_matrix2[i:i+8, j:j+8]
            x=cv2.matchTemplate(mask_matrix,sub_matrix1,cv2.TM_CCORR_NORMED)
            tmp1.append(x)
        tmp.append(tmp1)
    return tmp