#coding: utf-8 -*-
import sys

import locale

print(locale.getpreferredencoding())

import tensorflow as tf
import numpy as np
import random
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import common
pos_txt = "../data/tf_dat/pos.txt"
neg_txt = "../data/tf_dat/neg.txt"
#
# 1
# 2
# 3
#


def create_neural_network(data,net_config):
    layer_1_w_b = {'w_':tf.Variable(tf.random_normal([net_config['n_input_layer'],net_config['n_layer_1']])),
                   'b_':tf.Variable(tf.random_normal([net_config['n_layer_1']]))}

    layer_2_w_b = {'w_':tf.Variable(tf.random_normal([net_config['n_layer_1'], net_config['n_layer_2']])),
                   'b_':tf.Variable(tf.random_normal([net_config['n_layer_2']]))}

    layer_output_w_b = {'w_':tf.Variable(tf.random_normal([net_config['n_layer_2'],net_config['n_output_layer']])),
                        'b_':tf.Variable(tf.random_normal([net_config['n_output_layer']]))}

    print(data.shape)
    layer_1 = tf.add(tf.matmul(data, layer_1_w_b['w_']),layer_1_w_b['b_'])
    layer_1 = tf.nn.relu(layer_1)

    layer_2 = tf.add(tf.matmul(layer_1, layer_2_w_b['w_']),layer_2_w_b['b_'])
    layer_2 = tf.nn.relu(layer_2)
    layer_output = tf.add(tf.matmul(layer_2,layer_output_w_b['w_']),layer_output_w_b['b_'])
    return layer_output


def train_neural_network(X,Y,batch_size,train_data, test_data,net_config):
    predict = create_neural_network(X,net_config)
    cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict,labels=Y))
    optimizer = tf.train.AdamOptimizer().minimize(cost_func)
    epochs = 13
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        epoch_loss = 0
        i = 0
        random.shuffle(train_data)
        train_x = train_data[:,0]
        train_y = train_data[:,1]
        for epoch in range(epochs):
            while i < len(train_x):
                start = i
                end = i + batch_size
                batch_x = train_x[start:end]
                batch_y = train_y[start:end]
                _,c = sess.run([optimizer,cost_func],
                               feed_dict={X:list(batch_x),Y:list(batch_y)})
                epoch_loss += c
                i += batch_size

            print(epoch, ':', epoch_loss)

        test_x = test_data[:,0]
        test_y = test_data[:,1]
        correct = tf.equal(tf.argmax(predict,1),tf.argmax(Y,1))
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        print('accuracy: ', accuracy.eval({X:list(test_x), Y:list(test_y)}))


def main():
    lex = common.create_lexicon(pos_txt, neg_txt)

    dataset = common.normalize_dataset(lex, pos_txt, neg_txt)
    random.shuffle(dataset)
    print(len(dataset))

    test_size = int(len(dataset) * 0.1)
    dataset = np.array(dataset)

    train_data = dataset[:-test_size]
    test_data = dataset[-test_size:]

    net_conf = {}

    n_input_layer = len(lex)

    print("input layer number: %d"%n_input_layer)
    n_layer_1 = 2000
    n_layer_2 = 2000

    n_output_layer = 2
    net_conf['n_input_layer'] = n_input_layer
    net_conf['n_output_layer'] = n_output_layer
    net_conf['n_layer_1'] = n_layer_1
    net_conf['n_layer_2'] = n_layer_2
    batch_size = 50
    X = tf.placeholder('float', [None, len(train_data[0][0])])
    Y = tf.placeholder('float')
    train_neural_network(X, Y,batch_size,train_data,test_data,net_conf)


if __name__ == "__main__":
    main()


