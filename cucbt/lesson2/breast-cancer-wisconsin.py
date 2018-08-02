import numpy as np
import scipy
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score

# Number of samples (train/valid/test) : 699
# Number of features : 9
# Number of classes : 2
# Valid range of feature value: 1 - 10

def sigmoid(A):
    return 1 / (1 + np.exp(-A))

if __name__ == '__main__':
    with open('breast-cancer-wisconsin.data', 'r') as file:
        data_file = file.read().replace('?', '5').split('\n')
    # print (data_file)
    data1 = []
    for item in data_file:
        data1.append(item.split(','))

    data = []
    label = []

    for i in data1:
        temp = []
        count = 0
        # print len(i)
        for j in i:
            if(count == 10):
                if(int(j) == 2):
                    label.append(0)
                else:
                    label.append((1))
            elif (count != 0):
                temp.append(j)
            count += 1
        data.append(temp)
    data = [list(map(int, x)) for x in data]

    index = int (len(data)*0.8)
    index2 = index + int(len(data) *0.1)
    batch_size = 10000
    learning_rate = 0.0559
    Loss = []

    data_train = data[:index]
    data_test = data[index:index2]
    data_valid = data[index2:]
    label_train = label[:index]
    label_test = label[index:index2]
    label_valid = label[index2:]

    n = len(data_train)
    # y_train = y_train.reshape((num_samples, 1))
    # print(label_train)
    X = np.matrix(data_train)
    Y = np.matrix(label_train)
    Y = Y.transpose()
    W1 = np.random.rand(9, 10)
    W2 = np.random.rand(10, 1)

    for i in range (batch_size):
        Z1= np.dot(X, W1)
        L1= sigmoid(Z1)
        Z2= np.dot(L1, W2)
        y_hat = sigmoid(Z2)

        tmp1 = np.multiply((y_hat-Y), y_hat, (1-y_hat))
        tmp2 = np.multiply(y_hat - Y, y_hat, 1 - y_hat).dot(W2.transpose())
        tmp2 = np.multiply(tmp2, L1, 1-L1)
        dW1= (X.transpose().dot(tmp2))/n
        # dW1 = X.transpose().dot(np.multiply(np.multiply(y_hat - Y, y_hat, 1 - y_hat).dot(W2.transpose()), L1, 1 - L1)) / n
        # dW1 = (X.transpose().dot(tmp1))/n
        dW2 = (np.dot(L1.transpose(), tmp1))/n
        # dW1 = (X.transpose().dot(np.multiply((y_hat-Y), y_hat, (1-y_hat))).dot(np.multiply(W2.transpose(), L1, (1-L1))))/n
        # dW2 = (L1.transpose().dot(np.multiply(y_hat - Y, y_hat, 1 - y_hat))) / n

        W1 = W1 - learning_rate * dW1
        W2 = W2 - learning_rate * dW2

        l = np.multiply((y_hat - Y), (y_hat - Y)).sum() / (2 * n)
        Loss.append(l)

    X_test = np.matrix(data_test)
    y_hat_test = sigmoid(sigmoid(X_test.dot(W1)).dot(W2))
    # print(y_hat_test)
    label_pred = []
    for item in y_hat_test:
        if(item > 0.5):
            label_pred.append(1)
        else:
            label_pred.append(0)
    # print(type(label_train))
    acc = accuracy_score(label_test, label_pred)
    print('acc = {}'.format(acc))
    plt.plot(Loss)
    plt.show()