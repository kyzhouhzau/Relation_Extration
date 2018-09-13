#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:zhoukaiyin
参考：https://github.com/FrankWork/conv_relation
参考
"""
#第一步将训练数据和测试数据处理成如下格式
#18 1 1 9 9 the child was carefully wrapped and bound into the cradle by means of a cord
#第一个为label 第二个为<e1>起始位置，依次类推第四个为<e2>起始位置
#1.利用TEST_FILE_KEY.TXT文件编码label使得train和test中的label相对应,将编码结果写入data/label2id.pkl文件
import pickle
import re
import os

def checklabel(label2idname):
    with open(label2idname,'rb') as f:
        result = pickle.load(f)
        if len(result)==19:
            print(result)
            print("Success finish label to id mapping!")

def label2id(path,label2idname):
    if not os.path.exists(label2idname):
        with open(path) as tf:
            with open(label2idname,'wb') as wf:
                di = {}
                for line in tf:
                    line_list = line.strip().split(' ')
                    di[line_list[-1]]=int(line_list[0])
                pickle.dump(di,wf)
#将raw_data转成
#18 1 1 9 9 the child was carefully wrapped and bound into the cradle by means of a cord
#并且对数据进行清洗，去除特殊符号，和单个字符等
def load_labelid(label2idname):
    with open(label2idname,'rb') as rf:
        result = pickle.load(rf)
        return result

def raw2new(rawfile,newfile,label2idname):
    label2iddir = load_labelid(label2idname)
    patten = re.compile("(\d+	)")
    patten1 = re.compile('["():?.,!;\_\~\`]')
    pattene1 = re.compile(".*?(<e\d>)(.*?)(</e\d>).*?")
    patten2 = re.compile("(.*?)(</e\d>).*?")
    patten3 = re.compile(".*?(<e\d>)(.*)")
    stringdir={}

    rf=open(rawfile,'r')
    wf=open(newfile,'w')
    sentencelist = rf.readlines()
    for i,line in enumerate(sentencelist):
        line = line.lower()
        line_num  = patten.match(line)
        if line_num:
            label = label2iddir[sentencelist[i+1].strip()]
            line = re.sub(patten,"",line)
            n_line = re.sub(patten1,"",line)
            words = n_line.split()
            for i,word in enumerate(words):
                if pattene1.match(word):
                    if pattene1.search(word).group(1)=="<e1>":
                        stringdir["e11"]=i
                        stringdir["e12"] = i
                        words[i]=pattene1.search(word).group(2)
                    elif pattene1.search(word).group(1)=="<e2>":
                        stringdir["e21"] = i
                        stringdir["e22"] = i
                        words[i]=pattene1.search(word).group(2)
                elif patten2.match(word):
                    if patten2.search(word).group(2)=="</e1>":
                        stringdir["e12"] = i
                        words[i] = patten2.search(word).group(1)
                    elif patten2.search(word).group(2)=="</e2>":
                        stringdir["e22"] = i
                        words[i] = patten2.search(word).group(1)
                elif patten3.match(word):
                    if patten3.search(word).group(1)=="<e1>":
                        stringdir["e11"] = i
                        words[i] = patten3.search(word).group(2)
                    elif patten3.search(word).group(1)=="<e2>":
                        stringdir["e21"] = i

                        words[i] = patten3.search(word).group(2)
            sentence = ' '.join(words)
            new_line = "{} {} {} {} {} {}\n".format(label,stringdir["e11"],stringdir["e12"],stringdir["e21"],stringdir["e22"],sentence)
            wf.write(new_line)
    rf.close()
    wf.close()

if __name__=='__main__':
    # datatype="Test"
    label2idname = "../data/vocab/label2id.pkl"
    path = "../data/relations.txt"
    label2id(path,label2idname)
    trainrawfile = "../raw_data/TRAIN_FILE.TXT"
    trainnewfile = "../data/train"
    raw2new(trainrawfile,trainnewfile,label2idname)
    testrawfile = "../raw_data/TEST_FILE_FULL.TXT"
    testnewfile = "../data/test"
    raw2new(testrawfile,testnewfile,label2idname)
