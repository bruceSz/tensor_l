# -*- coding: utf-8 -*-
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import tensorflow as tf

import pickle
import numpy as np
import pandas as pd
import common

from collections import OrderedDict

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
    cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(predict,Y))
    optimizer = tf.train.AdamOptimizer().minimize(cost_func)

    with tf.Session() as s:
        s.run(tf.initialize_all_variables())
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




def main():
    preprocess()




if __name__ =="__main__":
    main()
