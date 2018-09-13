#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:zhoukaiyin
@Time:2018/8/26
"""
import tensorflow as tf
FLAGS = tf.app.flags.FLAGS
def linear_transformation(name,conv_output,in_size,out_size):
     with tf.variable_scope(name):
         w = tf.get_variable(name+"weight",shape=[in_size,out_size],
                             initializer=tf.truncated_normal_initializer(stddev=0.1))
         b = tf.get_variable(name+"bias",shape=[out_size],initializer=tf.constant_initializer(0.1))
         linear_out = tf.matmul(conv_output,w)+b
         #tensorflow 实战第二版上提供的误差添加方法。
         tf.add_to_collection("loss",tf.contrib.layers.l2_regularizer(FLAGS.l2_learning)(w))
         return linear_out

def cnn_layer(name,sentence,lexical,num_filter,linearoutput_dim=200):
    sentence_dim = sentence.shape[2]
    with tf.variable_scope(name):
        pool_outputs = []
        for filter_size in [2,3,4]:
            with tf.variable_scope("conv-%s" % filter_size):
                conv_weight = tf.get_variable("Weight",[filter_size,FLAGS.max_len,1,num_filter],
                                              initializer=tf.truncated_normal_initializer(0.1))
                bias = tf.get_variable("bias",[num_filter],initializer=tf.constant_initializer(0.1))
                n_sentence = tf.reshape(sentence,[-1,sentence_dim])
                linear = linear_transformation("before_conv_linear",n_sentence,sentence_dim,linearoutput_dim)
                linear_oup = tf.reshape(linear,[-1,FLAGS.max_len,linearoutput_dim])
                linear_oup = tf.expand_dims(linear_oup,axis=-1)
                conv = tf.nn.conv2d(linear_oup,conv_weight,strides=[1,1,linearoutput_dim,1],padding="SAME")
                conv = tf.nn.relu(conv)
                conv_output = tf.nn.bias_add(conv,bias)#sentencelen*3dim
                pool_average = tf.nn.avg_pool(conv_output,ksize=[1,FLAGS.max_len,1,1],strides=[1,FLAGS.max_len,1,1],padding="SAME")
                pool_max = tf.nn.max_pool(conv_output,ksize=[1,FLAGS.max_len,1,1],strides=[1,FLAGS.max_len,1,1],padding="SAME")
                pool_a = tf.get_variable("pool_a",[],initializer=tf.constant_initializer(0.5))
                pool_b = tf.get_variable("pool_b",[],initializer=tf.constant_initializer(0.5))
                pool = tf.add(tf.multiply(pool_max,pool_a),tf.multiply(pool_average,pool_b))
                pool_outputs.append(pool)
                pools = tf.concat(pool_outputs,3)
        n_pool = tf.reshape(pools,shape=[-1,3*num_filter])
        tanh_transformation = tf.nn.tanh(linear_transformation("tanh_w2",n_pool,3*num_filter,200))
        feature = tf.concat([lexical,tanh_transformation],axis=1)
        return feature

class Model(object):
    def __init__(self,word_embed,wordnet_embed,data,num_classes,word_dim,
                    keep_prob,num_filter,position_dim,is_train=True,regulizer = True):
        label, lexical,wordnet, position, sentence = data
        word_embed = tf.get_variable("word_embed",initializer=word_embed,
                                     dtype=tf.float32,trainable=True)
        wordnet_embed = tf.get_variable("wordnet_embed",initializer=wordnet_embed,
                                     dtype=tf.float32,trainable=True)
        lexicals = tf.nn.embedding_lookup(word_embed,lexical)
        wordnet = tf.nn.embedding_lookup(wordnet_embed,wordnet)
        possition_embed = tf.get_variable('possition_embed', shape=[2*FLAGS.embed_num, position_dim],
                                            initializer=tf.truncated_normal_initializer(0.1))
        sentences = tf.nn.embedding_lookup(word_embed,sentence)
        sentences = tf.reshape(sentences,[-1,FLAGS.max_len,3*word_dim])
        labelsss = tf.one_hot(label,num_classes)
        lexicalss = tf.reshape(lexicals,[-1,6*word_dim])
        # lexicals_ = tf.reshape(lexicals,[-1,6*word_dim])
        wordnet = tf.reshape(wordnet,[-1,2*word_dim])
        lexicals_ = tf.concat([lexicalss,wordnet],axis=1)

        position_ = tf.nn.embedding_lookup(possition_embed,position)
        position = tf.reshape(position_,[-1,FLAGS.max_len,2*position_dim])
        sentence_ = tf.concat([sentences,position], axis=2)
        if is_train:
            sentence_ = tf.nn.dropout(sentence_, keep_prob)
        feature = cnn_layer("feature",sentence_,lexicals_,num_filter)
        if is_train:
            feature = tf.nn.dropout(feature,keep_prob)
        feature_size = feature.shape[1]
        output = linear_transformation("output",feature,feature_size,num_classes)
        prediction = tf.nn.softmax(output,axis=1)#axis=1 表示按照纵轴
        label = tf.argmax(prediction,axis=1)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(label,tf.argmax(labelsss,axis=1)),tf.float32))
        loss_ = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = labelsss,logits=prediction))
        tf.add_to_collection("loss",loss_)
        loss = tf.add_n(tf.get_collection("loss"))
        # if regulizer:
        #     tv = tf.trainable_variables()
        #     regulization_loss = FLAGS.l2_learning*tf.add_n([tf.nn.l2_loss(v) for v in tv])
        # loss = loss_+regulization_loss
        self.label = label
        self.accuracy = accuracy
        self.loss = loss
        if not is_train:
            return
        global_step = tf.Variable(0,trainable=False,name='sep',dtype=tf.float32)
        #learning_rate = tf.train.exponential_decay(FLAGS.learningrate,global_step,100,0.96,staircase=True)
        optimizer = tf.train.AdamOptimizer(FLAGS.learningrate)
        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimizer = optimizer.minimize(loss,global_step)
        self.global_step = global_step

def train_or_valid(train_data,test_data,word_embed,wordnet_embed):
    with tf.name_scope("Train"):
        with tf.variable_scope("model",reuse=None):
            mtrain = Model(word_embed,wordnet_embed,train_data,FLAGS.num_classes,
                            FLAGS.word_dim,FLAGS.keep_prob,FLAGS.num_filter,position_dim=FLAGS.position_dim,is_train=True)
    
    with tf.name_scope("test"):   
        with tf.variable_scope("model",reuse=True):
            mtest = Model(word_embed,wordnet_embed,test_data,FLAGS.num_classes,
                            FLAGS.word_dim,num_filter=FLAGS.num_filter,keep_prob=1.0,position_dim=FLAGS.position_dim,is_train=False)
    return mtrain,mtest

class set_save(object):
    @classmethod
    def save_model(cls,sess,model_path):
        cls.saver = tf.train.Saver()
        cls.saver.save(sess,model_path)
    @classmethod
    def load_model(cls,sess,model_path):
        cls.saver = tf.train.Saver()
        cls.saver.restore(sess,model_path)

