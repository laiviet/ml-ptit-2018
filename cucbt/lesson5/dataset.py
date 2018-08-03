
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

def write_file(file_path, list, begin, end, len):
    file = open(file_path, 'w+')
    for i in range(begin, end):
        if(len[i] == 1):
            file.write(list[i])
            file.write("\n")
    file.close()

if __name__ == '__main__':
    data_file = "/home/cucbui/code/code/RNN/compression-data.json"

    ori_sentences = []
    com_sentences = []
    bin_sentences = []
    len = []

    with open (data_file) as f:
        data = f.read()

    count = 0
    for item in decode_stacked(data):
        bin = []
        begin = item['source_tree']['node'][1]['word'][0]['id']
        end = item['source_tree']['node'][-1]['word'][0]['id']
        l = end - begin + 1
        len.append(1) if l <= 50 else len.append(0)
        sentence = item['graph']['sentence']
        if(sentence[-1] != '.'):
            sentence += '.'
        ori_sentences.append(sentence)
        sentence = item['compression_untransformed']['text']
        if (sentence[-1] != '.'):
            sentence += '.'
        com_sentences.append(sentence)
        for i in range(l):
            bin.append('0')
        for i in item['compression_untransformed']['edge']:
            bin[i['child_id'] - begin] = '1'
        bin = ''.join(bin)
        bin_sentences.append(bin)

    write_file('train.ori', ori_sentences, 0, 8000, len)
    write_file('train.com', com_sentences, 0, 8000, len)
    write_file('train.bin', bin_sentences, 0, 8000, len)
    write_file('valid.ori', ori_sentences, 8000, 9000, len)
    write_file('valid.com', com_sentences, 8000, 9000, len)
    write_file('valid.bin', bin_sentences, 8000, 9000, len)
    write_file('test.ori', ori_sentences, 9000, 10000, len)
    write_file('test.com', com_sentences, 9000, 10000, len)
    write_file('test.bin', bin_sentences, 9000, 10000, len)