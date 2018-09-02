from __future__ import print_function

import json

def isNotStop(c):
    return c not in ['.' , '?' , '!']

def addJsonObj(str , cnt):
    if cnt <= 8000 :
        pathFile = './data/train'
    elif cnt <= 9000:
        pathFile = './data/valid'
    else:
        pathFile = './data/test'
    jso = json.loads(str)
    oriS ,comS = jso["source_tree"]["sentence"] , jso["compression_untransformed"]["text"]
    jsoA , jsoB = jso["source_tree"]["node"] , jso["compression_untransformed"]["edge"]
    mi , ma = 1000 , -1
    listW = []
    for a in jsoA:
        for b in a["word"]:
            if b["id"]!= -1 :
                listW.append( ( b["id"] , b["form"] ) )
                mi , ma = min (mi , b["id"]) , max (ma , b["id"])
    listW.sort(key=lambda tup : tup[0])
    if isNotStop(listW[len(listW) - 1][1]) :
        listW.append((ma+1 , "."))
    oriS = ""
    for a,b in listW:
        oriS = oriS + b + ' '
    oriS = oriS[:len(oriS) - 1]  + oriS[len(oriS):]

    mp = {}
    for a in jsoB :
        mp[a['child_id']] = 1
    mp[ma] = 1
    if (len (listW) > 50) : return

    binS = ""
    for a,b in listW:
        if mp.has_key(a):
            binS += '1'
        else :
            binS += '0'

    if (len(oriS.split(' ')) != len(binS)):
        print(mi , ma)
        print(oriS)
        print(comS)
        print(binS)
        print(len(oriS.split(' '))  , len(binS))

    print (oriS.encode('UTF-8') , file = open(pathFile + '.ori' , 'a') )
    print (binS.encode('UTF-8') , file = open(pathFile + '.bin' , 'a') )

f = open('./data/compression-data.json')
# f = open('./data/demo.json')
str = ""
cnt = 0
for line in f :
    for i in range(len(line)):
        if line[i]!='\n':
            if len(str) >0 and str[len(str) -1] == '}' and line[i] == '{':
                cnt += 1
                addJsonObj(str , cnt)
                str = ""
            str+=line[i]
cnt += 1
addJsonObj(str , cnt)