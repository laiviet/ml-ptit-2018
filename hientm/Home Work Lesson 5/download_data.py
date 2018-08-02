from json import JSONDecodeError, JSONDecoder
import os
import re
import requests

data_link = 'http://storage.googleapis.com/sentencecomp/compression-data.json'
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


def load_data(data_link):
    if not os.path.isfile('./data.json'):
        try:
            temp = requests.get(data_link).text
        except requests.RequestException:
            print("Error Conection!")
        with open('data.json', 'w') as f:
            f.write(temp)
    with open('data.json', 'r') as f:
        temp = f.read()
    return temp
