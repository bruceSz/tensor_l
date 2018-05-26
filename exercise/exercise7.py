#coding: utf-8

import collections
_P_FILES = "../data/poetry_utf8.txt"
import codecs
import numpy as np
import tensorflow as tf

def poem_reader():
    poetrys = []
    with codecs.open(_P_FILES,'r','utf-8') as f:
        for line in f:
            try:
                title,content = line.strip().split(":")

                content = content.replace(" ",'')
                content = content.replace("_","")
                #print(content)
                #poetrys.append(content)
                #if '_' in content or '(' in content or "《" in content or '[' in content:
                #    continue
                #if len(content) < 5 or len(content) > 79:
                #    continue
                #print(content)
                #content = '['+content+']'
                poetrys.append(content)
            except Exception as e:
                pass
    return poetrys


def poem_preprocess(words = 1000):
    poems = poem_reader()
    all_words = []
    for p in poems:
        all_words += [w for w in p ]
    counter = collections.Counter(all_words)
    counter_pairs = sorted(counter.items(),key=lambda x:-x[-1])
    counter_pairs = counter_pairs[:words]
    print(len(counter_pairs))
    words, _ = zip(*counter_pairs)
    words = words[:len(words)] + (' ',)
    word_num_map = dict(zip(words, range(len(words))))
    to_num = lambda word:word_num_map.get(word,len(words))
    poetrys_vectors = [list(map(to_num,p)) for p in poems]

    return poetrys_vectors,words,word_num_map
    #words= dict(zip(counter_pairs))
    #print(words)

def create_batches(poetrys_vecs,word_num_map):
    batch_size = 64
    n_chunk = len(poetrys_vecs) // batch_size
    x_batches = []
    y_batches = []
    for i in range(n_chunk):
        start_idx = i * (batch_size)
        end_index = start_idx + batch_size
        batches = poetrys_vecs[start_idx:end_index]
        length = max(map(len,batches))
        xdata = np.full((batch_size,length),word_num_map[' '],np.int32)
        for row in range(batch_size):
            xdata[row,:len(batches[row])] = batches[row]
        ydata = np.copy(xdata)
        ydata[:,:-1] = xdata[:,1:]
    x_batches.append(xdata)
    y_batches.append(ydata)
    return x_batches,y_batches

class PoetryNet(object):
    def __init__(self,model='lstm',rnn_size = 128, num_layers = 2, batch_size=64,
               words_num = 1000):
        self.model = model
        self.rnn_size = rnn_size
        self.batch_size = batch_size
        self.word_num = words_num
        self.num_layers = num_layers
        self.input_data = tf.placeholder(tf.int32,[batch_size,None])
        self.output_targets = tf.placeholder(tf.int32,[batch_size,None])

    def init_net_struct(self):

        if self.model == 'rnn':
            cell_fun = tf.nn.rnn_cell.BasicRNNCell
        elif self.model == 'gru':
            cell_fun = tf.nn.rnn_cell.GRUCell
        elif self.model == 'lstm':
            cell_fun = tf.nn.rnn_cell.BasicLSTMCell

        cell = cell_fun(self.rnn_size,state_is_tuple=True)
        # TODO.
        # To understood the multirnncell.

        cell = tf.nn.rnn_cell.MultiRNNCell([cell]*self.num_layers,state_is_tuple=True)

        initial_state = cell.zero_state(self.batch_size, tf.float32)

        with tf.variable_scope("rnnlm"):
            softmax_w = tf.get_variable("softmax_w",[self.rnn_size, self.word_num+1])
            softmax_b = tf.get_variable("softmax_b",[self.word_num+1])
            with tf.device("/cpu:0"):
                embedding = tf.get_variable("embedding",[self.word_num+1,self.rnn_size])
                inputs = tf.nn.embedding_lookup(embedding,self.input_data)
        outputs,last_state = tf.nn.dynamic_rnn(cell, inputs,
                                               initial_state=initial_state,
                                               scope='rnnlm')
        output = tf.reshape(outputs,[-1,self.rnn_size])
        logits = tf.matmul(output, softmax_w) + softmax_b
        probs = tf.nn.softmax(logits)
        self.logits =  logits#,\
        self.last_state  = last_state
        #,probs,cell,initial_state


    def train_neural_network(self):
        logits,last_state = self.logits,self.last_state
        batch_size = 64
        poetrys_vecs,words,word_num_map = poem_preprocess()
        x_batches, y_batches = create_batches(poetrys_vecs, word_num_map)
        output_targets = tf.placeholder(tf.int32,[batch_size,None])
        n_chunk = len(poetrys_vecs) // batch_size
        targets = tf.reshape(output_targets, [-1])
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits],[targets],[
            tf.ones_like(targets,dtype=tf.float32)
        ],1000)
        cost = tf.reduce_mean(loss)
        learning_rate = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost,tvars),5)
        opti = tf.train.AdamOptimizer(learning_rate)
        train_op = opti.apply_gradients(zip(grads,tvars))
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            saver = tf.train.Saver(tf.all_variables())
            for epoch in range(50):
                sess.run(tf.assign(learning_rate,0.002*(0.97**epoch)))
                n = 0
                for batche in range(n_chunk):
                    print x_batches[n].shape
                    train_loss,_,_ = sess.run([cost,last_state,train_op],
                                              feed_dict={self.input_data:x_batches[n],
                                                         output_targets:y_batches[n]})
                    n+= 1
                    print(epoch,batche, train_loss)
                if epoch % 7 == 0:
                    saver.save(sess,'poetry.module',global_step=epoch)


def main():
    #poem_preprocess()
    pn = PoetryNet()
    pn.init_net_struct()
    pn.train_neural_network()


if __name__ == "__main__":
    main()