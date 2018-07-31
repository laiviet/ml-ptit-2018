from json import JSONDecoder, JSONDecodeError
import re

file=open('D:\\compression-data.json','r').read()

def decode_stacked(document, pos=0, decoder=JSONDecoder()):
    NOT_WHITESPACE = re.compile(r'[^\s]')
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

def write_file(ori_file,com_file,bin_file,ori_sentence,com_sentence,bin_label):
    ori_file.write("%s\n" % origin_sentence)
    com_file.write("%s\n" % compresstion_sentence)
    for item in binary_label:
        bin_file.write("%s" % item)
    bin_file.write("\n")
train_origin = open('train.ori','w')
train_compresstion = open('train.com','w')
train_binary=open('train.bin','w')

valid_origin = open('valid.ori','w')
valid_compresstion = open('valid.com','w')
valid_binary=open('valid.bin','w')

test_origin = open('test.ori','w')
test_compresstion = open('test.com','w')
test_binary=open('test.bin','w')

train_count, valid_count, test_count=0,0,0
i=-1
for obj in decode_stacked(file):
    i+=1
    start_index=obj['source_tree']['edge'][0]['child_id']
    temp=obj['compression_untransformed']['edge']
    end_index=0
    index_list=[]
    for id in temp:
        index_list.append(id['child_id'])
        end_index=int(id['child_id'])
    if end_index-start_index+1>50:
        continue
    index_list=[x - start_index for x in index_list]
    binary_label=['0']*(end_index-start_index+1)
    origin_sentence=obj['graph']['sentence']
    compresstion_sentence=obj['compression_untransformed']['text']
    for j in index_list:
        binary_label[j]='1'
    if i in range(0,8000):
        train_count+=1
        write_file(ori_file=train_origin,com_file=train_compresstion,
                    bin_file=train_binary,ori_sentence=origin_sentence,
                    com_sentence=compresstion_sentence,bin_label=binary_label)
    elif i in range(8000,9000):
        valid_count+=1
        write_file(ori_file=valid_origin,com_file=valid_compresstion,
                    bin_file=valid_binary,ori_sentence=origin_sentence,
                    com_sentence=compresstion_sentence,bin_label=binary_label)
    elif i in range(9000,10000):
        test_count+=1
        write_file(ori_file=test_origin,com_file=test_compresstion,
                    bin_file=test_binary,ori_sentence=origin_sentence,
                    com_sentence=compresstion_sentence,bin_label=binary_label)

print(train_count,valid_count,test_count,i)

train_origin.close()
train_compresstion.close()
train_binary.close()
valid_origin.close()
valid_compresstion.close()
valid_binary.close()
test_origin.close()
test_compresstion.close()
test_binary.close()
