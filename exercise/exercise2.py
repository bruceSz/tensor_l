# -*- coding: utf-8 -*-
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import tensorflow as tf

import pickle
import numpy as np
import common


# 参考： https://zhuanlan.zhihu.com/p/28087321

org_train_file = "../data/trainingandtestdata/training.1600000.processed.noemoticon.csv"
org_test_file = "../data/trainingandtestdata/testdata.manual.2009.06.14.csv"

o_training_file = "../data/trainingandtestdata/training.csv"
o_testing_file = "../data/trainingandtestdata/testing.csv"
lexcion_f = "../data/trainingandtestdata/lexcion.pickle"

def preprocess():
    common.useful_filed(org_train_file, o_training_file)
    common.useful_filed(org_test_file, o_testing_file)
    lex = common.create_tweet_lexicon_buffer(o_training_file)
    print(len(lex))
    with open(lexcion_f, 'wb') as f:
        pickle.dump(lex, f)

def neural_2_layer_network(data,
                   n_input_layer,
                   n_layer_1,
                   n_layer_2,
                   n_output_layer):
    # first level neural w and b
    layer_1_w_b = {'w_':tf.Variable(tf.random_normal([n_input_layer,n_layer_1])),
                   'b_':tf.Variable(tf.random_normal([n_layer_1]))}
    layer_2_w_b = {'w_':tf.Variable(tf.random_normal([n_layer_1,n_layer_2])),
                   'b_':tf.Variable(tf.random_normal([n_layer_2]))}
    layer_output_w_b = {'w_':tf.Variable(tf.random_normal([n_layer_2,n_output_layer])),
                        'b_':tf.Variable(tf.random_normal([n_output_layer]))}

    layer_1 = tf.add(tf.matmul(data,layer_1_w_b['w_']),layer_1_w_b['b_'])
    layer_1  = tf.nn.relu(layer_1)
    layer_2 = tf.add(tf.matmul(layer_1,layer_2_w_b['w_']),layer_2_w_b['b_'])
    layer_2 = tf.nn.relu(layer_2)
    layer_output = tf.add(tf.matmul(layer_2,layer_output_w_b['w_']),layer_output_w_b['b_'])
    return layer_output



def neural_network_cnn(X,dropout_keep_prob,num_classes,input_size):
    with tf.device('/cpu:0'),tf.name_scope('embedding'):
        embedding_size= 128
        W = tf.Variable(tf.random_uniform([input_size,embedding_size],-1.0,1.0))
        embedded_chars = tf.nn.embedding_lookup(W,X)
        embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)
    num_filters = 128
    filter_sizes=[3,4,5]
    pooled_outputs  = []
    for i , filter_size in enumerate(filter_sizes):
        with tf.name_scope("conv-maxpool-%s"%filter_size):
            filter_shape = [filter_size,embedding_size,1,num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape,stddev=0.1))
            b = tf.Variable(tf.constant(0.1,shape=[num_filters]))
            conv = tf.nn.conv2d(embedded_chars_expanded,W,strides=[1,1,1,1],padding='VALID')
            h = tf.nn.relu(tf.nn.bias_add(conv,b))
            pooled = tf.nn.max_pool(h,ksize=[1,input_size-filter_size+1,1,1],strides=[1,1,1,1],padding='VALID')
            pooled_outputs.append(pooled)

    num_filters_total = num_filters*len(filter_sizes)
    h_pool = tf.concat(pooled_outputs,3)
    h_pool_flat = tf.reshape(h_pool,[-1,num_filters_total])

    with tf.name_scope("dropout"):
        h_drop = tf.nn.dropout(h_pool_flat,dropout_keep_prob)
    with tf.name_scope("output"):
        W = tf.get_variable("W",shape=[num_filters_total,num_classes],
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.constant(0.1,shape=[num_classes]))
        output = tf.nn.xw_plus_b(h_drop,W,b)

    return output


def train_with_cnn():

    with open(lexcion_f,'rb') as lex_f:
        lex = pickle.load(lex_f)

    test_x,test_y = common.get_test_dataset(o_testing_file,lex)
    print("type: %s"%type(test_x))
    input_size = len(lex)
    num_classes = 3
    X = tf.placeholder(tf.int32,[None,input_size])
    Y = tf.placeholder(tf.float32,[None,num_classes])
    dropout_keep_prob = tf.placeholder(tf.float32)
    batch_size = 90

    output = neural_network_cnn(X,dropout_keep_prob=dropout_keep_prob,
                                num_classes=num_classes,input_size=input_size)
    optimizer = tf.train.AdamOptimizer(1e-3)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output,labels=Y))
    grads_and_vars = optimizer.compute_gradients(loss)
    train_op = optimizer.apply_gradients(grads_and_vars)
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        lemmatizer = WordNetLemmatizer()
        i =0
        while True:
            batch_x = []
            batch_y = []
            try:
                print("type of o_training_file: %s"%o_training_file)
                lines = common.get_n_random_line(o_training_file, batch_size)
                for line in lines:
                    label = line.split(":%:%:%:")[0]
                    tweet = line.split(":%:%:%:")[1]
                    words = word_tokenize(tweet.lower())
                    words = [lemmatizer.lemmatize(w) for w in words]
                    fts = np.zeros(len(lex))
                    for w in words:
                        if w in lex:
                            fts[lex.index(w)] = 1
                    batch_x.append(list(fts))
                    batch_y.append(eval(label))
                _,loss_ = sess.run([train_op,loss],feed_dict={X:batch_x,Y:batch_y,dropout_keep_prob:0.5})
                print(loss_)
            except Exception as e:
                raise RuntimeError("xxxx")
                print(e)
            if i %10 ==0:
                predictions = tf.argmax(output,1)
                correct_predictions = tf.equal(predictions,tf.argmax(Y,1))
                accuracy = tf.reduce_mean(tf.cast(correct_predictions,"float"))
                accur = sess.run(accuracy,feed_dict={X:test_x[0:50],Y:test_y[0:50],dropout_keep_prob:1.0})
                print("准确率:",accur)
            i+=1




def train():
    with open(lexcion_f,'rb') as lex_f:
        lex = pickle.load(lex_f)

    test_x,test_y = common.get_test_dataset(o_testing_file,lex)
    n_input_layer = len(lex)
    n_layer_1 = 2000
    n_layer_2 = 2000
    n_output_layer = 3
    X = tf.placeholder('float')
    Y = tf.placeholder('float')
    batch_size = 90
    predict = neural_2_layer_network(X,n_input_layer,n_layer_1,n_layer_2,n_output_layer)
    cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict,labels=Y))
    optimizer = tf.train.AdamOptimizer().minimize(cost_func)

    with tf.Session() as s:
        s.run(tf.global_variables_initializer())
        lemmatizer = WordNetLemmatizer()
        saver = tf.train.Saver()
        i = 0
        pre_accuracy = 0
        while True:
            batch_x = []
            batch_y = []
            try:
                lines = common.get_n_random_line(o_training_file,batch_size)
                for line in lines:
                    label = line.split(":%:%:%:")[0]
                    tweet = line.split(":%:%:%:")[1]
                    words = word_tokenize(tweet.lower())
                    words = [lemmatizer.lemmatize(w) for w in words]
                    fts = np.zeros(len(lex))
                    for w in words:
                        if w in lex:
                            fts[lex.index(w)] = 1
                    batch_x.append(list(fts))
                    batch_y.append(eval(label))
                s.run([optimizer,cost_func],feed_dict={X:batch_x,Y:batch_y})
            except Exception as e:
                print(e)
            if i>100:
                correct = tf.equal(tf.argmax(predict,1),tf.argmax(Y,1))
                accuracy = tf.reduce_mean(tf.cast(correct,'float'))
                accuracy = accuracy.eval({X:test_x,Y:test_y})
                if accuracy > pre_accuracy:
                    print("准确率", accuracy)
                    pre_accuracy = accuracy
                    saver.save(s, '../data/model.ckpt')
                i = 0
            i+= 1

def test_predict(tweet,lex):
    X = tf.placeholder('float')
    predict = neural_2_layer_network(X)

    with tf.Session() as s:
        s.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(s,'model.ckpt')

        lemmatizer = WordNetLemmatizer()
        words = word_tokenize(tweet.lower())
        words = [lemmatizer.lemmatize(w) for w in words]
        fts = np.zeros(len(lex))
        for w in words:
            if w in lex:
                fts[lex.lex.index(w)]  = 1
        res = s.run(tf.argmax(predict.eval(feed_dict={X:[fts]}),1))
        return  res



def main():
    #preprocess()
    train()
    #train_with_cnn()




if __name__ =="__main__":
    main()
