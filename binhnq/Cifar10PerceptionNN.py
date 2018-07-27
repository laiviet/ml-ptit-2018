import cPickle
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import torch.nn.functional as F

# lrs = [0.09 , 0.01 , 0.009, 0.001]
lrs = [0,01]
idx_lr = 0
history_loss_train =[]
eps = 0.001

def unpickle(fileName):
    with open(fileName, 'rb') as f:
        dict = cPickle.load(f)
    return dict

def join_batch(number_batch=5):
    for i in range(number_batch):
        fileName = "./data/cifar-10-batches-py/data_batch_" +str(i+1)
        data = unpickle(fileName)
        if i == 0:
            features = data['data']
            labels = np.array(data["labels"])
        else:
            features = np.append(features , data['data'])
            labels = np.append(labels , data['labels'])
    return features,labels

def one_hot(data):
    one_hot = np.zeros((data.shape[0], 10))
    one_hot[np.arange(data.shape[0]), data] = 1
    return one_hot

def nomalize(data):
    return data/255.0

def getTrainData(number_batch=5):
    X, Y = join_batch(number_batch)
    X = nomalize(X)
    X = X.reshape(-1,3072)
    # Y = one_hot(Y)
    # Y = Y.reshape(-1,1)
    print str(X.shape) + " " + str(Y.shape)
    return X,Y

def getTestData():
    data = unpickle("./data/cifar-10-batches-py/test_batch")
    features = data['data']
    labels = np.array(data['labels'])

    return features ,labels

def random_batch(X , Y , batchsize):
    rand_id = np.random.randint( 0 , len(X) , batchsize)
    x_batch = X[rand_id]
    y_batch = Y[rand_id]
    return x_batch , y_batch

class MLP(nn.Module):
    def __init__(self, n_features, n_classes):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(n_features, 200)
        self.layer2 = nn.Linear(200, 150)
        self.layer3 = nn.Linear(150, n_classes)

    def forward(self, x, training=True):
        x = F.relu(self.layer1(x))
        x = F.dropout(x, 0.5, training=training)
        x = F.relu(self.layer2(x))
        x = F.dropout(x, 0.5, training=training)
        x = self.layer3(x)
        return x

X_train , Y_train = getTrainData(5)
X_test , Y_test = getTestData()

epoch = 5000

model = MLP(3072, 10)
optimizer = torch.optim.Adam(model.parameters() , lr=lrs[idx_lr])
criterion = nn.CrossEntropyLoss()

X_tr , Y_tr = X_train , Y_train


for e in range(epoch):
    X_tr , Y_tr = random_batch(X_train , Y_train , 128)
    features = Variable(torch.from_numpy(X_tr).type(torch.FloatTensor))
    target = Variable(torch.from_numpy(Y_tr))
    optimizer.zero_grad()
    output = model(features)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    history_loss_train.append(loss)

    if e % 10 == 0:
        _, y_pred = torch.max(output, 1)
        y_pred = y_pred.data.numpy()
        acc = accuracy_score(Y_tr, y_pred)
        print("Acc = %f" % (acc * 100))

    if e == 0:
        previous_loss = loss
    else:
        if abs(loss - previous_loss) <= eps :
            if idx_lr != 0:
                idx_lr -= 1

    previous_loss = loss
    if e % 1 == 0:
        print('Epoch %d: %f' % (e, loss))
print('DONE')

features = Variable(torch.from_numpy(X_test).type(torch.FloatTensor))

with torch.no_grad():
    correct = 0
    total = 0
    output = model(features)
    _, y_pred = torch.max(output, 1)
    y_pred = y_pred.data.numpy()
    acc = accuracy_score(Y_test, y_pred)
    print("Acc = %f" % (acc * 100))


# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')

plt.plot(history_loss_train)
plt.title('model loss train')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()