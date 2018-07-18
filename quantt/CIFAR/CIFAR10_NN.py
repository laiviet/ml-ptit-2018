import numpy as np
from sklearn.metrics import accuracy_score, f1_score,confusion_matrix, precision_recall_fscore_support
from torch.optim.lr_scheduler import MultiStepLR
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from LoadData import BatchManager

load=BatchManager()

epoch = 80000
lr = 0.0001

class ModuleA(nn.Module):
    def __init__(self):
        super(ModuleA,self).__init__()
        self.l1=nn.Linear(3072,1200)
        self.l2 = nn.Linear(1200, 600)
        self.sigmoid=nn.ReLU()
        self.l3=nn.Linear(600,10)
    def forward(self,x):
        x=self.l1(x)
        x=self.sigmoid(x)
        x=self.l2(x)
        x=self.sigmoid(x)
        x = self.l3(x)
        return x

model = ModuleA()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()
scheduler = MultiStepLR(optimizer, milestones=[60000], gamma=0.1)
y_hat=None
log = []
for e in range(epoch):
    data=load.gettrain()
    features = Variable(torch.from_numpy(data[0]).type(torch.FloatTensor))
    target = Variable(torch.from_numpy(data[1]).type(torch.LongTensor))
    y_hat = model(features)
    loss = criterion(y_hat, target)
    optimizer.zero_grad()
    loss.backward()
    scheduler.step()
    optimizer.step()
    if e % 100 == 0:
        _, y_pred = torch.max(y_hat,1)
        y_pred = y_pred.data.numpy()
        acc = accuracy_score(data[1], y_pred)
        string='Epoch %d: %f %f' % (e, loss,acc)
        if epoch%1000==0:
            log.append(string)
        print(string)

thefile = open('train_result.txt', 'w')
for item in log:
    thefile.write("%s\n" % item)
data=load.gettest()
features = Variable(torch.from_numpy(data[0]).type(torch.FloatTensor))
target = Variable(torch.from_numpy(data[1]).type(torch.LongTensor))
y_hat = model(features)
_, y_pred = torch.max(y_hat,1)
y_pred = y_pred.data.numpy()
acc = accuracy_score(data[1], y_pred)
p, r, f, _ = precision_recall_fscore_support(data[1], y_pred)
print(acc)
thefile = open('test_result.txt', 'w')
thefile.write("%s\n" % 'Accuracy: '+str(acc))
thefile.write("\n%s" % 'Precision: ')
thefile.write("\n ".join(map(str, p)))
thefile.write("\n%s" % 'Recall: ')
thefile.write("\n".join(map(str, r)))
thefile.write("\n%s" % 'Fscore: ')
thefile.write("\n".join(map(str, f)))
