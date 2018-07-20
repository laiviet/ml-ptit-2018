import numpy as np
from sklearn.metrics import accuracy_score, f1_score,confusion_matrix, precision_recall_fscore_support 
from sklearn.preprocessing import MinMaxScaler
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import os
import pickle

from sklearn.model_selection import train_test_split
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

data_url = "./data/cifar-10-batches-py/"
i = 0
list_data = []
list_label = []
for i in range(1,2):
    url = os.path.join(data_url,'data_batch_'+str(i))
    temp = unpickle(url)
    list1 = np.empty([10000,1024])
    for i in range(10000):
        for j in range(1024):
            list1[i][j] = (int(temp[b'data'][i][j])+int(temp[b'data'][i][j+1024])+int(temp[b'data'][i][j+2048]))/3
    list_data.append(list1)
    list_label.append(temp[b'labels'])
flat_list_data = [item for sublist in list_data for item in sublist]
flat_list_label = [item for sublist in list_label for item in sublist]
scaler = MinMaxScaler()
scaler.fit(flat_list_data)
flat_list_data = scaler.transform(flat_list_data)
X_train, X_test, y_train, y_test = train_test_split(flat_list_data, flat_list_label, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)
epoch = 12000
h = 100
lr = 0.01
model = nn.Sequential(
    nn.Linear(1024,h),
    nn.Sigmoid(),
    nn.Linear(h,h),
    nn.Sigmoid(),
    nn.Linear(h,10)
)
optimizer = torch.optim.SGD(model.parameters(), lr=lr)  
criterion = nn.CrossEntropyLoss()
features = Variable(torch.from_numpy(np.array(X_train)).type(torch.FloatTensor))
target = Variable(torch.from_numpy(np.array(y_train)))
log = []
for e in range(epoch):
    y_hat = model(features)
    loss = criterion(y_hat, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if e % 200 == 0:
        _, y_pred = torch.max(y_hat, 1)
        y_pred = y_pred.data.numpy()
        acc = accuracy_score(y_train, y_pred)
        p,r,f,_ = precision_recall_fscore_support(y_train, y_pred)
        #cm = confusion_matrix(y, y_pred)
        log.append((e, loss.data[0], f[0],f[1]))
    if e % 1000 == 0:
        print('Epoch %d: %f , acc = %f'%(e, loss,acc))
print('DONE')
