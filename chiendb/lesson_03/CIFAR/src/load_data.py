import cPickle
import numpy as np

class LoadData:
    def __init__(self):
        dir = '/home/chiendb/Data/cifar-10-batches-py/'
        self.X_train = self.unpickle(dir + 'data_batch_1')['data'] / 255
        self.X_test = self.unpickle(dir + 'test_batch')['data'] / 255
        self.y_train = self.unpickle(dir + 'data_batch_1')['labels']
        self.y_test = self.unpickle(dir + 'test_batch')['labels']
        self.index = 0
        self.batch_size = 128
        for i in range(2, 6):
            self.X_train += self.unpickle(dir + 'data_batch_' + str(i))['data']
            self.y_train += self.unpickle(dir + 'data_batch_' + str(i))['labels']

        self.size = len(self.y_test)
        # self.y_train = self.vector_to_one_hot(np.array(self.y_train), 10)

    def get_batch(self):
        if self.index + self.batch_size > self.size:
            self.index = 0
        r = np.array(self.X_train[self.index: self.index + self.batch_size, :]), np.array(
            self.y_train[self.index: self.index + self.batch_size])
        self.index += self.batch_size
        return r

    def get_test(self):
        return np.array(self.X_test), np.array(self.y_test)

    def vector_to_one_hot(self, vector, n_classes):
        vector = vector.astype(np.int32)
        m = np.zeros((vector.shape[0], n_classes))
        m[np.arange(vector.shape[0]), vector] = 1
        return m

    def unpickle(self, file):
        with open(file, 'rb') as fo:
            dict = cPickle.load(fo)
        return dict

