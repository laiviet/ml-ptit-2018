import csv
import numpy as np

FILE_NAME = '../data/wdbc.data'

def scaleVariable(X):
    n = len(X)
    m = len(X[0])
    maX = X.max(axis=0)
    miX = X.min(axis=0)
    for j in range(m): # each colum
        ma = maX[j]
        mi = miX[j]
        for i in range(n):
            X[i][j] = (X[i][j] - (ma + mi) /2) / (ma - mi)
    return X

def readFile( fileName ):
    # input csv_file return features n X_matrix and  n labels Y_vector
    file = open(fileName)
    sc = csv.reader(file)

    lRow = []
    for row in sc:
        lRow.append(row)
    np.random.seed(7)
    np.random.shuffle(lRow) # shuffle data

    n = len(lRow)
    m = len(lRow[0])

    X = np.zeros( (n , m-2) )
    Y = np.zeros( (n , 1) )

    id = 0
    for row in lRow:
        Y[id][0] = float(row[1] == 'M')
        for i in range(2, len(row)):
            X[id][i-2] = float(row[i])
        id+=1

    return ( scaleVariable(X) , Y)

def getData():
    (X,Y) = readFile(FILE_NAME)
    n = len(X)
    m = len(X[0])

    n_test  = int(n*0.2)
    n_valid = n_test
    n_train = n - n_test - n_valid

    X_train = np.zeros((n_train, m))
    Y_train = np.zeros((n_train, 1))
    for i in range(n_train):
        X_train[i] = X[i]
        Y_train[i] = Y[i]

    X_test  = np.zeros((n_test, m))
    Y_test  = np.zeros((n_test, 1))
    for i in range(n_test):
        X_test[i] = X[i + n_train]
        Y_test[i] = Y[i + n_train]

    X_valid  = np.zeros((n_valid, m))
    Y_valid  = np.zeros((n_valid, 1))
    for i in range(n_valid):
        X_valid[i] = X[i + n_train + n_test]
        Y_valid[i] = Y[i + n_train + n_test]

    ans = ((X_train , Y_train) , (X_test , Y_test) , (X_valid , Y_valid))
    return ans

