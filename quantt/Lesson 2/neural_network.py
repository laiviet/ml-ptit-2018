import numpy as np
from sklearn.metrics import accuracy_score, f1_score,confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

def load_data(path):
    file= open(path, 'r')
    lines = file.readlines()
    lines=[l.strip().replace('?','5').split(',') for l in lines]
    lines = np.array(lines)
    x = lines[:,1:-1].astype(np.float16)
    y = lines[:,-1].astype(np.int8)
    y = y/2-1
    y=y.astype(np.int8)
    y.astype(np.int8)
    onehot_encoder = OneHotEncoder(sparse=False)
    y=y.reshape(y.shape[0],1)
    y = onehot_encoder.fit_transform(y)
    return (x,y)
class two_layers_nn():
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))
    def softmax(self,x):
        x = x + 1e-5
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)
    def data_scaler(self,x):
        scaler = MinMaxScaler()
        scaler.fit(x)
        x = scaler.transform(x)
        return x
    def cross_entropy(self,yhat, y):
        return -np.sum(y*np.log(yhat))/y.shape[0]
    def delta_ce(self,y_hat, y):
        grad = (y_hat -y)/y.shape[0]
        return grad
    def delta_sigmoid(self,l):
        return l*(1-l)
    def train(self,loop):

        W1 = np.random.random((self.d,self.h)) -1
        W2 = np.random.random((self.h,self.c)) -1
        b1 = np.random.random((1,self.h)) -1
        b2 = np.random.random((1,self.c)) -1
        for e in range(loop):

            z1 = np.dot(self.X, W1)+b1
            l1 = self.sigmoid(z1)
            z2 = np.dot(l1, W2)+b2
            y_hat = self.softmax(z2)
            if e % 100 ==0:
                loss = self.cross_entropy(y_hat, self.Y)
                y_pred = np.argmax(y_hat, axis=1)
                labels=np.argmax(self.Y,axis=1)

                acc = accuracy_score(labels, y_pred)
                if acc>0.97:
                    self.lr=self.lr/10
                    print('up')
                f = f1_score(labels, y_pred)
                print('Epoch %d: loss=%f > acc=%0.4f > f0=%0.4f'%(e, loss, acc, f))
            e_z2 = self.delta_ce(y_hat, self.Y)
            delta_W2 = np.dot(l1.T, e_z2)
            db1 = np.sum(e_z2, axis = 1, keepdims = True)
            e_l1 = np.dot(e_z2, W2.T)
            e_z1 = e_l1 * self.delta_sigmoid(l1)
            delta_W1 = np.dot(self.X.T, e_z1)
            db2 = np.sum(e_z1, axis = 1, keepdims = True)
            W1 = W1 - self.lr * delta_W1
            W2 = W2 - self.lr * delta_W2
            b1=b1-self.lr*db1
            b2=b2-self.lr*db2
    def __init__(self,lr,hl):
        self.lr = lr
        self.h = hl
    def set_data(self,zip):
        self.X=zip[0]
        self.Y=zip[1]
        self.b,self.d=self.X.shape
        self.X=self.data_scaler(self.X)
        self.c=len(np.unique(self.Y))
model=two_layers_nn(lr=0.0003,hl=100)
model.set_data(load_data(path = 'breast-cancer-wisconsin.data'))
model.train(50000)
