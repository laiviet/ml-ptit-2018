import numpy as np
import sys
import pickle
import torch.utils.data as data


class CIFAR10(data.Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    # root = './cifar-10-batches-py/'
    root = '/home/student/hientm_folder/jupyter/data/cifar-10-batches-py/'
    train_list = ["data_batch_1",
                  "data_batch_2",
                  "data_batch_3",
                  "data_batch_4"]

    valid_list = ["data_batch_5"]

    test_list = ["test_batch"]

    def __init__(self, type = 0, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.type = type  # training set or test set

        # now load the picked numpy arrays
        if self.type == 0:
            self.train_data = []
            self.train_labels = []
            for f in self.train_list:
                file = self.root + f
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
            self.train_data = self.train_data.reshape((40000, 3, 32, 32))  #b-c-h-w
            #self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC
        else:
            if self.type == 1:
                f = self.test_list[0]
                file = self.root + f
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
            else:
                f = self.valid_list[0]
                file = self.root + f
                fo = open(file, 'rb')
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding='latin1')
                self.valid_data = entry['data']
                if 'labels' in entry:
                    self.valid_labels = entry['labels']
                else:
                    self.valid_labels = entry['fine_labels']
                fo.close()
                self.valid_data = self.valid_data.reshape((10000, 3, 32, 32))

    def __getitem__(self, index):
        if self.type == 0:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            if self.type == 1:
                img, target = self.test_data[index], self.test_labels[index]
            else:
                img, target = self.valid_data[index], self.valid_labels[index]
        return img, target


    def __len__(self):
        if self.type == 0:
            return len(self.train_data)
        else:
            if (self.type == 1):
                return len(self.test_data)
            else:
                return len(self.valid_data)