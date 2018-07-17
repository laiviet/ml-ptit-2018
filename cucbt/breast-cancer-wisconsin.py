import numpy as np
import scipy
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score

# Number of samples (train/valid/test) : 699
# Number of features : 9
# Number of classes : 2
# Valid range of feature value: 1 - 10


def parse_to_list(str):
    result = []
    tmp = []
    l = len(str)
    i = 0
    index = 0
    temp = ""
    while (i < l):
        if(str[i] != ","):
            temp+=str[i]
            i += 1
        else:
            index += 1
            if(index == 1):
                temp = ""
                i+=1
            else:
                temp = int (temp)
                tmp.append(temp)
                temp = ""
                i+=1
    result.append(tmp)
    if(int(temp) == 2):
        result.append(0)
    else:
        result.append(1)
    return result

def load_data():
    data = []
    filepath = 'breast-cancer-wisconsin.data'
    try:
        with open(filepath) as fp:
            line = fp.readline()
            cnt = 1
            while line:
                line = fp.readline()
                tmp = parse_to_list(line)
                data.append(tmp)
                cnt += 1
    except Exception as e:
        return data

def sigmoid(A):
    return 1 / (1 + np.exp(-A))

if __name__ == '__main__':
    print("============================")
    print("Number of samples: 699")
    print ("Number of features : 9")
    print ("Number of classes : 2")
    print("Valid range of feature value: 1 - 10")
    print("============================")

    number_test = 699
    number_classes = 2
    count = 0
    learning_rate = 0.1
    loss = []
    label_true = []
    label_test = []
    data = load_data()
    W1 = np.random.randint(100, size=(9, 4))
    W2 = np.random.randint(100, size=(4, 1))
    # W1 = np.random.rand(9, 4)
    # W2 = np.random.rand(4, 1)
    # print(W1)
    # print(W2)
    for item in data:
        count += 1
        print('Test set number {}: {}'.format(count, item[0]))

        X = np.matrix(item[0])
        Z1 = X.dot(W1)
        L1 = sigmoid(Z1)
        Z2 = L1.dot(W2)
        y_hat = sigmoid(Z2)

        dW1 = (X.transpose().dot(np.multiply((y_hat-item[1]), y_hat, (1-y_hat))).dot(np.multiply(W2.transpose(), L1, (1-L1))))/number_test
        dW2 = (L1.transpose().dot(np.multiply(y_hat - item[1], y_hat, 1 - y_hat)))/number_test

        W1 = W1 - learning_rate * dW1
        W2 = W2 - learning_rate * dW2
        label_true.append(item[1])
        l_test = (sigmoid(sigmoid(X.dot(W1)).dot(W2)))
        if(l_test < 0.5):
            label_test.append(0)
        else:
            label_test.append(1)

        l = ((y_hat - item[1]) * (y_hat - item[1])).sum() / (2*number_test)
        acc=0
        print('Loss = {}'.format(l))
        loss.append(l)

        if(count == number_test):
            break
    acc = accuracy_score(label_true, label_test)

    print('acc = {}'.format(acc))
    plt.plot(loss)
    plt.show()