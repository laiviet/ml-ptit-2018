import numpy as np
import os
import pickle

class BatchManager():
    def __init__(self):
        self.trainZip=list()
        path = 'cifar-10-python/cifar-10-batches-py'
        temp = self.unpickle(os.path.join(path, 'data_batch_' + str(1)))
        self.trainZip.append(np.array(temp[b'data']))
        self.trainZip.append(np.array(temp[b'labels']))
        for i in range(2, 6):
            temp=self.unpickle(os.path.join(path, 'data_batch_'+str(i)))
            x=np.array(temp[b'data'])
            y=np.array(temp[b'labels'])
            self.trainZip[0]=np.vstack((self.trainZip[0],x))
            self.trainZip[1] = np.hstack((self.trainZip[1], y))
        temp = self.unpickle(os.path.join(path, 'test_batch'))
        x = np.array(temp[b'data'])
        y = np.array(temp[b'labels'])
        self.test=[x, y]
        self.index=0
    def gettrain(self):
        batchsize=128
        lastIndex=self.index
        if self.index>(self.trainZip[0].shape[0]-batchsize-1):
            self.index=batchsize-self.trainZip[0].shape[0]+1+self.index
            hx1=self.trainZip[0][lastIndex:,:]
            hx2=self.trainZip[0][:self.index,:]
            hy1=self.trainZip[1][lastIndex:]
            hy2=self.trainZip[1][:self.index]
            return [np.vstack((hx1,hx2)),np.hstack((hy1,hy2))]
        else:
            self.index+=batchsize
            return [self.trainZip[0][lastIndex:self.index,:],self.trainZip[1][lastIndex:self.index]]

    def gettest(self):
        return self.test
    def getall(self):
        return self.trainZip
    def unpickle(self,file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

