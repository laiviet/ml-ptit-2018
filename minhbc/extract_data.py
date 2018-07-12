import numpy as np
import sklearn
import pandas as pd 
import matplotlib.pyplot as plt

def extract():
    with open('./breast-cancer-wisconsin.data','r') as f:
        inputx = []
        outputy = []
        for line in f:
            temp = []
            temp3 = []
            i = 0
            for x in line.strip().split(','):
                if x == '?':
                    x = 0
                x1 = int(x)
                i+=1
                if i == 11:
                    if x1 == 2 :
                        temp3.append([1,0])
                    else:
                        temp3.append([0,1])
                    i = 0
                else:
                    temp.append(x1)
            inputx.append(temp)
            outputy.append(temp3)
        inputx = np.array(inputx)
        outputy = np.array(outputy)
    inputx = inputx.astype(float)
    total = np.sum(inputx,axis=0)
    xmax = np.amax(inputx,axis=0)
    for i in range(inputx.shape[0]):
        for j in range(inputx.shape[1]):
            if j == 0 :
                continue
            inputx[i][j] = float((inputx[i][j]-(total[j]/inputx.shape[0]))/(xmax[j]-1))
    return inputx,outputy

    