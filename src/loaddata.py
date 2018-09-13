#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Time:2018/8/25
@Author:zhoukaiyin
"""
import numpy as np
import pickle
import gensim
import os
from nltk.corpus import wordnet
import tensorflow as tf
from collections import namedtuple

FLAGS = tf.flags.FLAGS

Data_example = namedtuple("data_example", "label,entity1,entity2,sentence")
Entity_position = namedtuple("Entity_position", "first last")



def load_raw_data(datapath):
    """
    将data的每一行数据以Data_example的格式保存。
    :return: data
    """
    data = []
    with open(datapath,'r') as rf:
        for line in rf:
            line_lis = line.strip().split()
            if len(line_lis)>FLAGS.max_len:
                FLAGS.max_len=len(line_lis)
            sentence = line_lis[5:]
            entity1 = Entity_position(int(line_lis[1]),int(line_lis[2]))
            entity2 = Entity_position(int(line_lis[3]),int(line_lis[4]))
            line_example = Data_example(line_lis[0],entity1,entity2,sentence)
            data.append(line_example)
    return data

def vocab2id(train_data,test_data,vocabfile):
    """
    将训练数据与测试数据中所有出现的单词去重后映射到id
    :param train_data:
    :param test_data:
    :param vocabfile:
    :return: 将vocab2id的映射写入磁盘
    """
    vocab=[]
    for example in train_data+test_data:
        for w in example.sentence:
            vocab.append(w)
    words = list(set(vocab))
    vocab2id = {w:i for i,w in enumerate(words,1)}
    vocab2id["<pad>"]=0
    if not os.path.exists(vocabfile):
        with open(vocabfile,'wb') as wf:
            pickle.dump(vocab2id,wf)

def _load_vocab(vocabfile):
    with open(vocabfile,'rb') as rf:
        vocab2id=pickle.load(rf)
    return vocab2id

def embed_trim(vocabfile,embedtrimfile):
    """
    由于词向量本身较大，这里通过修剪将不必要的去掉，只保留与vocabfile中对应的部分。
    :param vocabfile:
    :param embedtrimfile:google pre_trained word2vec
    :return:将修建后的向量存入磁盘供后面model的调用。
    """
    trimedembed=[]
    if not os.path.exists(embedtrimfile):
        if FLAGS.word_dim==200:
            model = gensim.models.KeyedVectors.load_word2vec_format("../data/embed/glove200.txt")
        elif FLAGS.word_dim == 50:
            model = gensim.models.KeyedVectors.load_word2vec_format("../data/embed/glove50.txt")
        #pad_embed =[0]*model.vector_size
        vocab2id = _load_vocab(vocabfile)
        id2vocab = {i:w for w,i in vocab2id.items()}
        ebed_vocab = model.vocab.keys()
        count=0
        for i in range(len(id2vocab)):
            w= id2vocab[i]
            if w in ebed_vocab:
                trimedembed.append(model[w])
            else:
                word_lis = w.split('_')
                try:
                    for m in word_lis:
                        np_embed = np.zeros([model.vector_size])
                        w_embed = model[m]
                        np_embed+=w_embed
                        trimedembed.append(list(np_embed))
                except KeyError:
                    count+=1
                    npdata = np.random.normal(0,0.1,[model.vector_size])
                    trimedembed.append(list(npdata))
        embed = np.asarray(trimedembed).astype(np.float32)
        print("在构建---单词---词向量的时候有{}个单词没有找到词向量!".format(count))
        np.save(embedtrimfile,embed)

def hypernym_embed(vocabfile,hypernymembedfile):
    def _get_hypernym(entity):
        try:
            nentity = wordnet.synsets(entity)
            h_entity = nentity[0].hypernyms()
            word = h_entity[0].name().split('.')[0]
        except Exception:
            # print("{} entity dont have hyperny word!".format(entity))
            word = entity
        return word

    hypernymembed=[]
    if not os.path.exists(hypernymembedfile):
        if FLAGS.word_dim==200:
            model = gensim.models.KeyedVectors.load_word2vec_format("../data/embed/glove200.txt")
        elif FLAGS.word_dim == 50:
            model = gensim.models.KeyedVectors.load_word2vec_format("../data/embed/glove50.txt")
        #pad_embed =[0]*model.vector_size
        vocab2id = _load_vocab(vocabfile)
        id2vocab = {i:w for w,i in vocab2id.items()}
        ebed_vocab = model.vocab.keys()
        count=0
        for i in range(len(id2vocab)):
            w= id2vocab[i]
            hypernym = _get_hypernym(w)
            if hypernym in ebed_vocab:
                hypernymembed.append(model[hypernym])
            else:
                word_lis = hypernym.split('_')
                try:
                    for k in word_lis:
                        np_embed = np.zeros([model.vector_size])
                        w_embed = model[k]
                        np_embed+=w_embed
                        hypernymembed.append(list(np_embed))
                except KeyError:
                    count+=1
                    npdata = np.random.normal(0,0.1,[model.vector_size])
                    hypernymembed.append(list(npdata))
        embed = np.asarray(hypernymembed).astype(np.float32)
        print("在构建---上位词---词向量的时候有{}个单词没有找到词向量!".format(count))
        np.save(hypernymembedfile,embed)


def build_hypernym_feature(data_example):
    hypernym1 = data_example.entity1[0]
    hypernym2 = data_example.entity2[0]
    hypernym = [hypernym1,hypernym2]
    return hypernym

def _load_embed(embedtrimfile):
    embed = np.load(embedtrimfile).astype(np.float32)
    return embed

def map_data2id(data_example,vocabfile):
    vocabid = _load_vocab(vocabfile)
    sentence = data_example.sentence
    for i,w in enumerate(sentence):
        m = vocabid[w]
        data_example.sentence[i]=m
    sen_len = len(data_example.sentence)
    if sen_len<FLAGS.max_len:
        data_example.sentence.extend([0]*(FLAGS.max_len-sen_len))

#build fiture :lexical level features
def _lexical(data_example):
    #这里思考由于有两个entity如何用定义两个列表分别对两个列表做同样的工作
    #显得累赘，于是将处理两个entity的过程写成函数
    def _entity_context(entity_id_first,entity_id_last,sentence):
        context=[]
        if entity_id_first==entity_id_last:
            context.append(sentence[entity_id_first])
            entity_id_last=entity_id_first
            if entity_id_first>=1:
                context.append(sentence[entity_id_first-1])
            else:
                context.append(sentence[entity_id_first])
            if entity_id_last<len(sentence)-1:
                context.append(sentence[entity_id_last+1])
            else:
                context.append(sentence[entity_id_last])
        else:
            context.append(sentence[entity_id_first])
            context.append(sentence[entity_id_first-1])
            context.append(sentence[entity_id_first+1])
        return context
    entity_1_first = data_example.entity1.first
    entity_1_last = data_example.entity1.last
    entity_2_first = data_example.entity2.first
    entity_2_last = data_example.entity2.last
    context1 = _entity_context(entity_1_first,entity_1_last,data_example.sentence)
    context2 = _entity_context(entity_2_first,entity_2_last,data_example.sentence)
    context = context1+context2
    return context

def _word_feature(data_example):
    """
    文章中将WF特征表示为[x_s,x_0,x_1],[x_0,x_1,x_2],[x_1,x_2,x_3],....(此处由于按照论文的编码结果不理想所以采用该种编码)
    但实际上仅仅以该单词的词向量作为WF特征就已经很合适，所以下面会给出两种WF的方案。
    :return:
    """
    sentence = data_example.sentence
    wordfeature = []
    for i,w in enumerate(sentence):
        WF=[]
        if i>0 and i<len(sentence)-1:
            WF.append(sentence[i-1])
            WF.append(sentence[i])
            WF.append(sentence[i+1])
        elif i==0:
            WF.append(sentence[i])
            WF.append(sentence[i])
            WF.append(sentence[i+1])
        else:
            WF.append(sentence[i])
            WF.append(sentence[i-1])
            WF.append(sentence[i-1])
        wordfeature.append(WF)
        # data_example.sentence[i] = w
    for i,newword in enumerate(wordfeature):
        data_example.sentence[i]=newword


def _position_feature(data_example):
    #位置特征是为了充分考虑entity特征，此外也可以考虑依存树特征。
    def _get_position(n):
        if n < -60:
            return 0
        elif n >= -60 and n <= 60:
            return n + 61
        return 122
    entity_1_first = data_example.entity1.first
    entity_2_first = data_example.entity2.first
    position1=[]
    position2=[]
    length = len(data_example.sentence)
    for i in range(length):
        position1.append(_get_position(i-entity_1_first))
        position2.append(_get_position(i-entity_2_first))
    position  = []
    for i,pos in enumerate(position2):
        position.append([pos,position1[i]])

    # def _get_position(entity_id_first,entity_id_last,sentence):
    #     position=[]
    #     if entity_id_first==entity_id_last:
    #         for i,w in enumerate(sentence):
    #             distance = i-entity_id_first
    #             position.append(distance+FLAGS.max_len)
    #         return position
    #     else:
    #         for i,w in enumerate(sentence):
    #             if i<=entity_id_first:
    #                 distance = i-entity_id_first
    #                 position.append(distance+FLAGS.max_len)
    #             elif i>entity_id_first and i<=entity_id_last:
    #                 distance = 0
    #                 position.append(distance+FLAGS.max_len)
    #             else:
    #                 distance = i-entity_id_last
    #                 position.append(distance+FLAGS.max_len)
    #         return position
    # entity_1_first = data_example.entity1.first
    # entity_1_last = data_example.entity1.last
    # entity_2_first = data_example.entity2.first
    # entity_2_last = data_example.entity2.last
    # position1 = _get_position(entity_1_first, entity_1_last, data_example.sentence)
    # position2 = _get_position(entity_2_first, entity_2_last, data_example.sentence)
    # FLAGS.embed_num=FLAGS.max_len
    
    return position

def build_sequence_example(data_example):
    """
    用tf.train.SequenceExample()函数将之前所做的特征存储起来。
    context 来放置非序列化部分；如：lexical，label(对于一个实例而言label是一个非序列化的数据)
    feature_lists 放置变长序列。如：WF,PF
    :param data_example:
    :return:example
    """
    lexical = _lexical(data_example)
    position = _position_feature(data_example)
    wordnet_feature = build_hypernym_feature(data_example)
    _word_feature(data_example)
    example = tf.train.SequenceExample()
    example.context.feature["lexical"].int64_list.value.extend(lexical)
    example.context.feature["wordnet"].int64_list.value.extend(wordnet_feature)
    example_label = int(data_example.label)
    example.context.feature["label"].int64_list.value.append(example_label)
    sentence = data_example.sentence
    for w in sentence:
        example.feature_lists.feature_list["sentence"].feature.add().int64_list.value.extend(w)
    for p in position:
        example.feature_lists.feature_list["position"].feature.add().int64_list.value.extend(p)

        

    return example

def tfrecord_write(data,tfrecordfilename):
    """
    将最初的data数据实例化成data_example数据再写入内存。
    :param data:
    :param filename:
    :return:
    """
    if not os.path.exists(tfrecordfilename):
        with tf.python_io.TFRecordWriter(tfrecordfilename) as writer:
            for data_example in data:
                map_data2id(data_example, FLAGS.vocabfile)
                example = build_sequence_example(data_example)
                writer.write(example.SerializeToString())

def parse_tfrecord(sereialized_example):
    context_features = {
        "lexical" : tf.FixedLenFeature([6],tf.int64),
        "label" : tf.FixedLenFeature([],tf.int64),
        "wordnet": tf.FixedLenFeature([2],tf.int64)
    }
    sequence_features = {
        "sentence" : tf.FixedLenSequenceFeature([3],tf.int64),
        "position" : tf.FixedLenSequenceFeature([2],tf.int64),
    }

    contex_dict,sequence_dic = tf.parse_single_sequence_example(sereialized_example,
    context_features=context_features,sequence_features=sequence_features)
    sentence = sequence_dic["sentence"]
    position = sequence_dic["position"]
    lexical = contex_dict["lexical"]
    wordnet = contex_dict["wordnet"]
    label = contex_dict["label"]
    return label,lexical,wordnet,position,sentence

def read_data_as_batch(tfrecordfilename,epoch,batchsize,shuffle=True):
    serized_data = tf.data.TFRecordDataset(tfrecordfilename)
    serized_data = serized_data.map(parse_tfrecord)
    serized_data = serized_data.repeat(epoch)
    if shuffle:
        serized_data = serized_data.shuffle(buffer_size=1000)
    serized_data = serized_data.batch(batchsize)
    iterator = serized_data.make_one_shot_iterator()
    batch = iterator.get_next()
    return batch

def inputs():

    train_data = load_raw_data(FLAGS.train_file)
    test_data = load_raw_data(FLAGS.test_file)
    vocab2id(train_data,test_data,FLAGS.vocabfile)
    embed_trim(FLAGS.vocabfile,FLAGS.embedtrimfile)
    hypernym_embed(FLAGS.vocabfile,FLAGS.hypernymembedfile)
    embed = _load_embed(FLAGS.embedtrimfile)
    wordnet_embed = _load_embed(FLAGS.hypernymembedfile)
    tfrecord_write(train_data, FLAGS.tfrecordfilename_train)
    tfrecord_write(test_data, FLAGS.tfrecordfilename_test)
    train = read_data_as_batch(FLAGS.tfrecordfilename_train,FLAGS.epoch,FLAGS.batchsize)
    test = read_data_as_batch(FLAGS.tfrecordfilename_test,FLAGS.epoch,batchsize=2717,shuffle=False)
    return train,test,embed,wordnet_embed

def write(label2idpath,test_resultpath,rediction_label):
    with open(label2idpath,'rb') as rf:
        with open(test_resultpath,'w') as wf:
            label2id_dir = pickle.load(rf)
            id2labeldir = {i:label for label,i in label2id_dir.items()}
            for i,relation in enumerate(rediction_label):
                wf.write("{}\t{}\n".format(i+8001,id2labeldir[relation]))

if __name__=="__main__":
    """
    测试：
    """
    flags = tf.app.flags
    flags.DEFINE_string("train_file","../data/train","train_file")
    flags.DEFINE_string("test_file","../data/test","test_file")
    flags.DEFINE_string("vocabfile","../data/vocab/vocabfile.pkl","vocabfile")
    flags.DEFINE_string("embedtrimfile","../data/embedtrimfile.npy","embedtrimfile")
    flags.DEFINE_string("position_init_embed","../data/position_init_embed.npy","position_init_embed")
    flags.DEFINE_string("tfrecordfilename_train","../data/tfrecord/tfrecordfilename_train","tfrecordfilename_train")
    flags.DEFINE_string("tfrecordfilename_test","../data/tfrecord/tfrecordfilename_test","tfrecordfilename_test")
    flags.DEFINE_integer("epoch",10,"epoch")
    flags.DEFINE_integer("embed_num",123,"embed_num")
    flags.DEFINE_integer("max_len",10,"max_len")
    flags.DEFINE_integer("batchsize",10,"batchsize")
    with tf.Session() as sess:
        train, test, embed= inputs()
        try:
            while True:
                print(sess.run(train))
                
        except tf.errors.OutOfRangeError:
            print("end!")




