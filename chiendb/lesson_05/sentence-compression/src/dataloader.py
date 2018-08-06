import json
import numpy as np


class DataLoader:
    def __init__(self):
        self.dir_path = '/home/chiendb/Data/sentence-compression-data/'
        self.seq_len = 100
        self.input_dim = 100
        self.dictionary = {}
        self.word_matrix = []
        with open('/home/chiendb/Data/glove.6B/glove.6B.100d.txt', 'r') as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            word = line.split()
            self.dictionary[word[0]] = i
            self.word_matrix.append([float(word[i]) for i in range(1, len(word))])

    def write_file(self, file_path, arr):
        f = open(file_path, 'w')
        for line in arr:
            f.write('{}\n'.format(line))

    def read_file(self, file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()

        return lines

    def crawler(self):
        with open(self.dir_path + 'compression-data.json', 'r') as f:
            text = f.readlines()

        lines = ''
        text.append('\n')
        sentences = []
        compressions = []
        bins = []
        for line in text:
            if line != '\n':
                lines += line
            else:
                js = json.loads(lines)
                node = js['source_tree']['node'][1:]
                sentence_dict = {word['word'][0]['id']: word['word'][0]['stem'] for word in node}
                sentence = []
                compression = js['compression_untransformed']['text']
                bin = []
                compression_id = [word['child_id'] for word in js['compression_untransformed']['edge']]
                for k, v in sentence_dict.items():
                    sentence.append(v)
                    if k in compression_id:
                        bin.append('1')
                    else:
                        bin.append('0')
                bin = ' '.join(bin)
                sentence = ' '.join(sentence)
                sentences.append(sentence)
                compressions.append(compression)
                bins.append(bin)
                lines = ''

        self.write_file(self.dir_path + 'train.ori', sentences[:8000])
        self.write_file(self.dir_path + 'valid.ori', sentences[8000:9000])
        self.write_file(self.dir_path + 'test.ori', sentences[9000:])
        self.write_file(self.dir_path + 'train.com', compressions[:8000])
        self.write_file(self.dir_path + 'valid.com', compressions[8000:9000])
        self.write_file(self.dir_path + 'test.com', compressions[9000:])
        self.write_file(self.dir_path + 'train.bin', bins[:8000])
        self.write_file(self.dir_path + 'valid.bin', bins[8000:9000])
        self.write_file(self.dir_path + 'test.bin', bins[9000:])

    def word2vec(self, lines):
        X = np.zeros((len(lines), self.seq_len, self.input_dim))
        for i, line in enumerate(lines):
            words = line.split()
            num_words = len(words)
            for j, word in enumerate(words):
                X[i][self.seq_len-num_words+j] = self.word_matrix[self.dictionary[word]]

        return X

    def str2int(self, lines):
        Y = np.zeros((len(lines), self.seq_len))
        for i, line in enumerate(lines):
            num = len(line.split())
            for j, v in enumerate(line.split()):
                Y[i][self.seq_len-num+j] = v

        return Y

    def get_data(self):
        features_train = self.word2vec(self.read_file(self.dir_path + 'train.ori'))
        target_train = self.str2int(self.read_file(self.dir_path + 'train.bin'))
        features_valid = self.word2vec(self.read_file(self.dir_path + 'valid.ori'))
        target_valid = self.str2int(self.read_file(self.dir_path + 'valid.bin'))
        features_test = self.word2vec(self.read_file(self.dir_path + 'test.ori'))
        target_test = self.str2int(self.read_file(self.dir_path + 'test.bin'))
        return {'feutures': features_train, 'target': target_train}, \
               {'feutures': features_valid, 'target': target_valid}, \
               {'feutures': features_test, 'target': target_test}


data_loader = DataLoader()
# data_loader.crawler()