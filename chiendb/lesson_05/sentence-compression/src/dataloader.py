import json


class DataLoader:
    def __init__(self, dir_path):
        self.dir_path = dir_path

    def write_file(self, file_path, arr):
        f = open(file_path, 'w')
        for line in arr:
            f.write('{}\n'.format(line))

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
                sentence = js['graph']['sentence']
                compression = js['compression_untransformed']['text']
                bin = []
                sentence_id = []
                node = js['graph']['node'][1:]
                for words in node:
                    sentence_id += [word['id'] for word in words['word']]
                compression_id = [word['child_id'] for word in js['compression_untransformed']['edge']]
                for i in range(min(sentence_id), max(sentence_id) + 1):
                    if i in compression_id:
                        bin.append('1')
                    else:
                        bin.append('0')
                bin = ' '.join(bin)
                if sentence[-1] != '.':
                    sentence += '.'
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


data_loader = DataLoader('/home/chiendb/Data/sentence-compression-data/')
data_loader.crawler()