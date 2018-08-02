
import json
from json import JSONDecoder, JSONDecodeError
import re

NOT_WHITESPACE = re.compile(r'[^\s]')

def decode_stacked(document, pos=0, decoder=JSONDecoder()):
    while True:
        match = NOT_WHITESPACE.search(document, pos)
        if not match:
            return
        pos = match.start()

        try:
            obj, pos = decoder.raw_decode(document, pos)
        except JSONDecodeError:
            raise
        yield obj

def write_file(file_path, list, begin, end):
    file = open(file_path, 'w+')
    for i in range(begin, end):
        file.write(list[i])
        file.write("\n")
    file.close()

if __name__ == '__main__':
    data_file = "/home/cucbui/code/code/RNN/compression-data.json"

    ori_sentences = []
    com_sentences = []
    bin_sentences = []

    with open (data_file) as f:
        data = f.read()

    count = 0
    for item in decode_stacked(data):

        sentence = item['graph']['sentence']
        if(sentence[-1] != '.'):
            sentence += '.'
        ori_sentences.append(sentence)
        sentence = item['compression_untransformed']['text']
        if (sentence[-1] != '.'):
            sentence += '.'
        com_sentences.append(sentence)

        sentence_id = []
        sentence = item['graph']['node'][1:]
        for word in sentence:
            sentence_id += [word['id'] for i in word['word']]
        compression_id = [word['child_id'] for i in word['compression_untransformed']['edge']]
        for i in range(min(sentence_id), max(sentence_id) + 1):
            if i in compression_id:
                bin.append('1')
            else:
                bin.append('0')
        bin = ' '.join(bin)
        bin_sentences.append(bin)

    write_file('train.ori', ori_sentences, 0, 8000)
    write_file('train.com', com_sentences, 0, 8000)
    write_file('train.bin', com_sentences, 0, 8000)
    write_file('valid.ori', ori_sentences, 8000, 9000)
    write_file('valid.com', com_sentences, 8000, 9000)
    write_file('valid.bin', com_sentences, 8000, 9000)
    write_file('test.ori', ori_sentences, 9000, 10000)
    write_file('test.com', com_sentences, 9000, 10000)
    write_file('test.bin', com_sentences, 9000, 10000)