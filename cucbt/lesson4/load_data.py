from torch.utils.data import Dataset
import numpy as np
import sys

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


class CIFAR10(Dataset):
    """ CIFAR 10 dataset."""
    train_list = [
        ['data_batch_1'],
        ['data_batch_2'],
        ['data_batch_3'],
        ['data_batch_4'],
    ]
    valid_list = [
        ['data_batch_5'],
    ]

    test_list = [
        ['test_batch'],
    ]
    data_path = "/home/cucbui/code/code/CIFAR10/cifar-10-batches-py/"
    # data_path ="/home/student/hientm_folder/jupyter/data/cifar-10-batches-py/"

    def __init__(self, data_type, transform=None):
        self.transform = transform
        self.data_type = data_type
        self.transform = transform

        if(self.data_type == 'train'):
            self.train_data = []
            self.train_labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file = self.data_path + f
                fo = open(file, 'rb')
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding='latin1')
                self.train_data.append(entry['data'])
                if 'labels' in entry:
                    self.train_labels += entry['labels']
                else:
                    self.train_labels += entry['fine_labels']
                fo.close()

            self.train_data = np.concatenate(self.train_data)
            self.train_data = self.train_data.reshape((40000, 3, 32, 32))

        elif(self.data_type=='valid'):
            self.valid_data = []
            self.valid_labels = []
            for fentry in self.valid_list:
                f = fentry[0]
                file = self.data_path + f
                fo = open(file, 'rb')
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding='latin1')
                self.valid_data.append(entry['data'])
                if 'labels' in entry:
                    self.valid_labels += entry['labels']
                else:
                    self.valid_labels += entry['fine_labels']
                fo.close()

            self.valid_data = np.concatenate(self.valid_data)
            self.valid_data = self.valid_data.reshape((10000, 3, 32, 32))
        else:
            f = self.test_list[0][0]
            file = self.data_path + f
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.test_data = entry['data']
            if 'labels' in entry:
                self.test_labels = entry['labels']
            else:
                self.test_labels = entry['fine_labels']
            fo.close()
            self.test_data = self.test_data.reshape((10000, 3, 32, 32))
        # self._load_meta()

    def __len__(self):
        if self.data_type == 'train':
            return len(self.train_data)
        elif self.data_type == 'test':
            return len(self.test_data)
        else:
            return len(self.valid_data)

    def __getitem__(self, index):
        if self.data_type == 'train':
            img, target = self.train_data[index], self.train_labels[index]
        elif self.data_type == 'test':
            img, target = self.test_data[index], self.test_labels[index]
        else:
            img, target = self.valid_data[index], self.valid_labels[index]

        return img, target


