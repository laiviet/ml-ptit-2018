import json
import numpy as np
import torch.utils.data as data


class Loader(data.Dataset):
    def __init__(self, type):
        self.dir_path = '/home/student/chiendb/data/'
        self.seq_len = 50
        self.input_dim = 100
        self.type = type

        if self.type == 0:
            self.features = self.read_file(self.dir_path + 'train.ori')
            self.target = self.read_file(self.dir_path + 'train.bin')
        elif self.type == 1:
            self.features = self.read_file(self.dir_path + 'valid.ori')
            self.target = self.read_file(self.dir_path + 'valid.bin')
        else:
            self.features = self.read_file(self.dir_path + 'test.ori')
            self.target = self.read_file(self.dir_path + 'test.bin')

        target_ = []
        for y_ in self.target:
            y = y_.split()
            len_ = len(y)
            target_.append([y[i] if i < len_ else 0 for i in range(self.seq_len)])

        self.target = target_

    def write_file(self, file_path, arr):
        f = open(file_path, 'w')
        for line in arr:
            f.write('{}\n'.format(line))

        f.close()

    def read_file(self, file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()

        return [line.replace('\n', '') for line in lines]

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

    def __getitem__(self, index):
        return self.features[index], self.target[index]

    def __len__(self):
        return len(self.features)

