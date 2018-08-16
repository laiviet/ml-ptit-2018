import download_data

data = download_data.load_data(download_data.data_link)


def write(ori, com, bin, name, start, end):
    temp = name + ".ori"
    with open(temp, 'w') as f:
        for i in range(start, end):
            f.write(ori[i])
            f.write("\n")
    temp = name + ".com"
    with open(temp, 'w') as f:
        for i in range(start, end):
            f.write(com[i])
            f.write("\n")
    temp = name + ".bin"
    with open(temp, 'w') as f:
        for i in range(start, end):
            f.write(bin[i])
            f.write("\n")


def write_data():
    ori_sentences = []
    com_sentences = []
    bin_sentences = []
    count = 0
    train_size = 0
    valid_size = 0
    test_size = 0
    for obj in download_data.decode_stacked(data):
        count += 1
        index_start = obj['source_tree']['node'][1]['word'][0]['id']
        index_end = obj['source_tree']['node'][-1]['word'][0]['id']
        temp1 = obj['source_tree']['node']
        sentence = []
        for word in temp1[1:]:
            sentence.append(word['form'])
        if len(sentence) > 50:
            continue
        temp = list('0' for i in range(len(sentence)))
        if sentence[-1] != '.':
            sentence.append('.')
            temp.append('1')
        sentence1 = ' '.join(sentence)
        ori_sentences.append(sentence1)
        com = obj['compression_untransformed']['text']
        if com[-1] != '.':
            com += '.'
        com_sentences.append(com)

        for i in obj['compression_untransformed']['edge']:
            if i['child_id'] - index_start < len(temp):
                temp[i['child_id'] - index_start] = '1'
        bin_sentences.append(''.join(temp))
        if count == 8000:
            train_size = len(ori_sentences)
        elif count == 9000:
            valid_size = len(ori_sentences)
        elif count == 10000:
            test_size = len(ori_sentences)

    write(ori_sentences, com_sentences, bin_sentences, "train", 0, train_size)
    write(ori_sentences, com_sentences, bin_sentences, "valid", train_size, valid_size)
    write(ori_sentences, com_sentences, bin_sentences, "test", valid_size, test_size)
