import cv2
import numpy
import numpy as np
import pywt
import math
def sumMaskMatrix(matrix):
    sum=0
    for i in range (0,8,1):
        for j in range(0, 8, 1):
          sum+=(matrix[i][j]*matrix[i][j])
    return sum

def calc_corr(noise_matrix1,noise_matrix2):
    size=noise_matrix1.shape
    col_size = size[1]
    row_size=size[0]
    sub_matrix1=numpy.zeros((8, 8))
    mask_matrix = numpy.zeros((8, 8))
    sub_matrix_withZeros1 = numpy.zeros((15, 15))
    result_matrix=numpy.zeros((row_size, col_size))
    row = 0
    col = 0
    for i in range (0,row_size,8):
        for j in range(0, col_size, 8):
            sub_matrix1=noise_matrix1[i:i+8,j:j+8]
            mask_matrix = noise_matrix2[i:i+8, j:j+8]
            sub_matrix_withZeros1[4:12,4:12]=sub_matrix1
            maskMatrixSum=sumMaskMatrix(mask_matrix)

            for x in range(0, 8, 1):
                for y in range(0, 8, 1):
                    cal_matrix = sub_matrix_withZeros1[x:x+8,y:y+8]
                    matrixSum = sumMaskMatrix(cal_matrix)
                    product=mask_matrix*cal_matrix
                    numerator=np.sum(product)
                    denominator=math.sqrt(maskMatrixSum*matrixSum)
                    result = numerator / denominator
                    result_matrix[row][col]=result
                    col=col+1
                    if(col==col_size):
                        row=row+1
                        col=0
    return result_matrix