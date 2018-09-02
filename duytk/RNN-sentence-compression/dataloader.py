import numpy as np
import sys
from tqdm import tqdm

import torch.utils.data as data
from enum import Enum
from itertools import izip
from gensim.models import KeyedVectors



class SC_DATA(data.Dataset):
    PATH_FILE = "./data/"
    w2v_file = './word2vec/word2vec.txt'

    # input : Batch x Sentence_Size x Vector_Dim
    # output: Batch x Sentence_Size x [0|1]

    def __init__(self , type , transform=None, target_transform=None):
        super(SC_DATA , self).__init__()
        self.transform = transform
        self.target_transform = target_transform
        ##############################

        model = KeyedVectors.load_word2vec_format(self.w2v_file)

        self.type = type
        path_file = self.PATH_FILE + type
        fori , fbin = open(path_file + ".ori") , open(path_file + ".bin")
        self.lSize = 50
        tmp = []
        for lori , lbin in  tqdm(izip (fori , fbin) , desc='DataLoading', leave=False):
            lw , lb = [] , []
            for w in lori.split(' '):
                if (w == '.\n'): w = '.'
                if w not in model.wv.vocab:
                    w = '<unk>'
                lw.append( model.wv[w])
            while (len(lw) < 50):
                lw.append( model.wv['<unk>'])
            for c in lbin:
                if c!= '\n': lb.append(ord(c) - ord('0'))
            while (len(lb) < 50):
                lb.append(0)
            tmp.append( (lw , lb , len(lb)) )

        self.inputs, self.labels, self.sizies = [], [], []
        for (lw , lb , le) in  tqdm(tmp , desc='DataLoading' , leave=False):
            self.inputs.append(np.array(lw))
            self.labels.append(np.array(lb))
            self.sizies.append(le)

        self.inputs = np.array(self.inputs)
        self.labels = np.array(self.labels)
        self.sizies = np.array(self.sizies)
        del model

    def __getitem__(self, index):
        return self.inputs[index] , self.labels[index] , self.sizies[index]

    def __len__(self):
        return len(self.inputs)