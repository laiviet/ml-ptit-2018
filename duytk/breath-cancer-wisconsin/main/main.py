import readerCSV as lib_reader
import numpy as np
import calculator as lib_calc
import matplotlib.pyplot as plt

# n - number of rows
# m - number of collums
# Number of samples : 569
# Number of features: 30
# Number of classes : 2
#

TRAIN = True

def getOutput(X,W1,W2):
    layer0 = X  # input_layer  30 features x 457 samples : 457x30
    layer1 = lib_calc.sigmoid(np.dot(layer0, W1))  # hidden_layer h = 15 neurals x n = 457 samples : 457x15
    layer2 = lib_calc.sigmoid(np.dot(layer1, W2))
    return layer2

def Build():
    (dataTrain , dataTest , dataValid ) = lib_reader.getData()
    X_train, Y_train = dataTrain
    X_test, Y_test = dataTest
    X_valid, Y_valid = dataValid

    print("================= Data-set-information ===============")
    print(" Number of dataTrain samples : ",len(Y_train))
    print(" Number of dataTest  samples : ",len(Y_test))
    print(" Number of dataValid samples : ",len(Y_valid))
    print(" Number of features          : ",len(X_train[0]) )
    print(" Number of classes           :  2")
    print("======================================================")

    # init model :w1 , w2, nEpochs = ??
    alpha = 0.001
    hSize = 32
    np.random.seed(7)
    W1 = np.random.random((X_train.shape[1] , hSize)) #dxh 30x15
    W2 = np.random.random((hSize , 1))                #hx1 15x1
    nEpochs = 10000
    historyTrain = []
    historyValid = []
    historyTest  = []

    # ==========================================TRAINING=======================================
    for epoch in range(nEpochs):
        layer0 = X_train                                   # input_layer  30 features x 457 samples : 457x30
        layer1 = lib_calc.sigmoid ( np.dot(layer0 , W1))   # hidden_layer h = 15 neurals x n = 457 samples : 457x15
        layer2 = lib_calc.sigmoid ( np.dot(layer1 , W2))   # output_layer c = 1 value x  n = 457 samples   : 457x1

        # caculate loss + test --> plot
        if (epoch%100 == 0):
            print("Epoch ",epoch, ":  ==============")
            historyTrain.append( lib_calc.caculate_and_show("training-set   :", layer2, Y_train)[0])
            historyValid.append( lib_calc.caculate_and_show("validation-set :", getOutput(X_valid , W1,W2), Y_valid)[0])
            historyTest.append( lib_calc.caculate_and_show("test-set       :", getOutput(X_test  , W1,W2), Y_test)[0])
            print("========================")



        # update model
        layer2_error = layer2 - Y_train
        layer2_delta = layer2_error * lib_calc.sigmoid_derivative(layer2)
        W2_delta     = layer1.T.dot(layer2_delta)

        layer1_error = layer2_delta.dot(W2.T)
        layer1_delta = layer1_error * lib_calc.sigmoid_derivative(layer1)
        W1_delta     = layer0.T.dot(layer1_delta)

        W1 -= alpha * W1_delta
        W2 -= alpha * W2_delta

    # plot to graph
    plt.plot(historyTrain)
    plt.plot(historyTest)
    plt.plot(historyValid)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test' , 'valid'], loc='upper left')
    plt.show()

if __name__ == '__main__':
    Build()