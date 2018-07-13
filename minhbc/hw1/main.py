import numpy as np 
from sklearn.cross_validation import train_test_split
import extract_data
import math
from sklearn.metrics import accuracy_score , f1_score, confusion_matrix
from matplotlib import pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

def subtract( mA, mB ) :
	# zero matrix
	result = [[0 for y in range(len(mA[0]))] for x in range(len(mA))]
	
	for i in range(len(mA)):
		for j in range(len(mA[i])):
			result[i][j] = mA[i][j]-mB[i][j]
	return result


learning_rate = 0.1
input_data,output_data = extract_data.extract()
input_data = np.delete(input_data,0, axis=1)
train_x,test_x,train_y,test_y = train_test_split(input_data,output_data,test_size = 0.33,random_state = 42)
w1 = np.random.rand(9,4)
w2 = np.random.rand(4,1)
total_loss = []
for i in range(10000):
    z1 = np.matmul(train_x,w1)
    l1 = sigmoid(z1)
    z2 = np.matmul(l1,w2)
    y_hat = sigmoid(z2)
    dw1 = np.matmul(np.transpose(train_x),np.multiply(np.multiply(np.matmul(np.multiply(np.multiply(np.true_divide((y_hat-train_y),(train_y.size/2)),y_hat),(1-y_hat)),np.transpose(w2)),l1),(1-l1)))
    dw2 = np.matmul(np.transpose(l1),np.multiply(np.multiply(np.true_divide((y_hat-train_y),(train_y.size/2)),y_hat),(1-y_hat)))
    w1 = w1 - learning_rate*dw1
    w2 = w2 - learning_rate*dw2
    z1 = np.matmul(test_x,w1)
    l1 = sigmoid(z1)
    z2 = np.matmul(l1,w2)
    y_hat = sigmoid(z2)
    for item in y_hat:
            if(item[0]>0.5):
                item[0]=1
            else:
                item[0]=0
    loss = np.sum(np.square((y_hat-test_y)))/test_x.shape[0]
    total_loss.append(loss)
    accuracy = accuracy_score(y_hat,test_y)
    f = f1_score(y_hat, test_y)
    # for i in range(test_x.shape[0]):
    #     y = y_hat[i].astype(int)
    #     if(np.array_equal(y,test_y[i][0])):
    #         count+=1
    # print("Accuracy : {}".format(count/test_x.shape[0]))
    if(i%100 == 0):
        print("Epoche {0}".format(i))
        # for i in range(test_x.shape[0]):
        #     if test_y[i][0] == 0:
        #         loss += -math.log(y_hat[i][0])
        #     else:
        #         loss+= -math.log(1-y_hat[i][0])
        # print(w1)
        # print("\n")
        # print(w2)
        print("Accuracy : {0} , f = {1} , loss = {2}".format(accuracy,f,loss))
plt.plot(total_loss)
plt.show()


