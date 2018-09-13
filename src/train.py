#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:zhoukaiyin
"""
import tensorflow as tf
from models import model
import loaddata
import os
flags = tf.app.flags
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
flags.DEFINE_string("train_file","../data/train","train_file")
flags.DEFINE_string("test_file","../data/test","test_file")
flags.DEFINE_string("vocabfile","../data/vocab/vocabfile.pkl","vocabfile")
flags.DEFINE_string("embedtrimfile","../data/embed/embedtrimfile.npy","embedtrimfile")
flags.DEFINE_string("hypernymembedfile","../data/embed/hypernymembedfile.npy","hypernymembedfile")
flags.DEFINE_string("tfrecordfilename_train","../data/tfrecord/tfrecordfilename_train","tfrecordfilename_train")
flags.DEFINE_string("tfrecordfilename_test","../data/tfrecord/tfrecordfilename_test","tfrecordfilename_test")
flags.DEFINE_string("test_result","../eval/result.txt","test_result")
flags.DEFINE_string("label2idpath","../data/vocab/label2id.pkl","label2idpath")
flags.DEFINE_string("model_path","../model/model.ckpt","model_path")
flags.DEFINE_integer("embed_num",123,"embed_num")
flags.DEFINE_integer("position_dim",5,"position_dim")
flags.DEFINE_integer("num_classes",19,"num_classes")
flags.DEFINE_integer("epoch",800,"epoch")
flags.DEFINE_integer("max_len",60,"max_len")
flags.DEFINE_integer("batchsize",100,"batchsize")
flags.DEFINE_integer("word_dim",50,"word_dim")
flags.DEFINE_integer("num_filter",100,"num_filter")
flags.DEFINE_float("l2_learning",0.001,"l2_learning")
flags.DEFINE_float("learningrate",0.001,"learningrate")
flags.DEFINE_float("keep_prob",0.6,"keep_prob")
flags.DEFINE_bool("train",True,"train")

FLAGS = tf.app.flags.FLAGS
def train(sess,mtrain,mtest):
    count = 0
    best=0
    while True:
        try:
            loss_,accuracy_,_= sess.run([mtrain.loss,mtrain.accuracy,mtrain.optimizer])
            if count%50==0:
                print("EPOCH:{}\tAccuracy:{:.4f}\t Loss:{:.4f}".format(count,accuracy_,loss_))
            if count%500==0 and count!=0:
                print("\n")
                print("Evalulating the result! Waitting !!!!!!!!\n")
                loss_test,accuracy_test = sess.run([mtest.loss,mtest.accuracy])
                if best < accuracy_test:
                    best = accuracy_test
                    model.set_save.save_model(sess, FLAGS.model_path)
                print("EPOCH:{}\tAccuracy:{:.4f}\t Loss:{:.4f}".format(count, accuracy_test, loss_test))
                print("\n")
            count+=1
        except tf.errors.OutOfRangeError:
            break

def test(sess,mtest):
    model.set_save.load_model(sess,FLAGS.model_path)
    accuracy_,prediction= sess.run([mtest.accuracy,mtest.label])
    loaddata.write(FLAGS.label2idpath,FLAGS.test_result,prediction)
    print("Accuracy:{:.4f}\t".format(accuracy_))

def main(_):
    with tf.Graph().as_default():
        train_data, test_data, word_embed,wordnet_embed = loaddata.inputs()
        mtrain,mtest = model.train_or_valid(train_data,test_data,word_embed,wordnet_embed)
        init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
        config = tf.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction=0.9
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(init_op)
            if FLAGS.train:
                train(sess,mtrain,mtest)
            else:
                test(sess,mtest)
                
if __name__=="__main__":
    tf.app.run()
