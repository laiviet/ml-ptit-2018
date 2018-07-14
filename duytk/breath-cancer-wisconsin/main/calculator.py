import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def loss(Y_hat , Y): #nx1
    ans = np.sum(np.square(Y_hat - Y)) / (2*len(Y_hat))
    return ans

def caculate_and_show(s , Y_hat , Y):
    # loss , acc , F1score
    myLoss = loss(Y_hat , Y)
    myAccuracy = accuracy_score(Y,(Y_hat >= 0.5))
    myF1Score = f1_score(Y,(Y_hat >= 0.5))
    print(s,': loss=%0.4f > acc=%0.4f > f1=%0.4f' % (myLoss, myAccuracy, myF1Score))
    return myLoss , myAccuracy , myF1Score

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)