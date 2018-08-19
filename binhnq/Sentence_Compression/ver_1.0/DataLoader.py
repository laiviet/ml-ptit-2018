import numpy as np
import torch.utils.data as data
import bcolz
class SC(data.Dataset):
    ROOT_PATH = "./SCdata/"

    def __init__(self,type_data, sentence_len , glove , transform = None):
        super(SC, self).__init__()
        self.transform = transform
        self.sentence_len = sentence_len
        self.inputs = []
        self.labels = []

        DATA_PATH = self.ROOT_PATH + type_data + '/'
        ori_file = open(DATA_PATH + type_data + ".ori")
        bin_file = open(DATA_PATH + type_data + ".bin")

        dictionary = dict()
        for word in glove:
            dictionary[word] = 1

        # np.random.normal(scale=0.6, size=(self.emb_dim,))

        list_bin_sentence = bin_file.readlines()
        list_ori_sentence = ori_file.readlines()
        print("Read data " + type_data +" done..")
        for index in range(0 , len(list_ori_sentence)):
            list_word , list_bin = [] , []

            for word in list_ori_sentence[index].replace(".\n",".").split(' '):
                if word not in dictionary:
                    word = '<unk>'
                list_word.append(glove[word])
                if (len(list_word) == sentence_len):
                    break

            while (len(list_word) < sentence_len):
                list_word.append(glove['<unk>'])
            for i in range(len(list_bin_sentence[index])):
                if len(list_bin) == sentence_len:
                    break
                if list_bin_sentence[index][i] == '0':
                    list_bin.append(0)
                else:
                    if list_bin_sentence[index][i] == '1':
                        list_bin.append(1)
            while (len(list_bin) < sentence_len):
                list_bin.append(0)

            self.inputs.append(np.array(list_word))
            self.labels.append(np.array(list_bin))


    def __getitem__(self, index):
        return self.inputs[index] , self.labels[index]

    def __len__(self):
        return len(self.inputs)
