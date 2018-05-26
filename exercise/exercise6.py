import numpy as np
import tensorflow as tf

import pandas as pd
from sklearn.preprocessing import scale
#https://archive.ics.uci.edu/ml/datasets/UJIIndoorLoc

_TRAIN = "../data/UJIndoorLoc/trainingData.csv"
_TEST = "../data/UJIndoorLoc/validationData.csv"



def encoder_tw_nn(X):
    # encoder
    e_w_1 = tf.Variable(tf.truncated_normal([520, 256], stddev=0.1))
    e_b_1 = tf.Variable(tf.constant(0.0, shape=[256]))
    e_w_2 = tf.Variable(tf.truncated_normal([256, 128], stddev=0.1))
    e_b_2 = tf.Variable(tf.constant(0.0, shape=[128]))
    e_w_3 = tf.Variable(tf.truncated_normal([128, 64], stddev=0.1))
    e_b_3 = tf.Variable(tf.constant(0.0, shape=[64]))


    # decoder
    # tied weight.
    d_w_1 = tf.transpose(e_w_3)
    d_b_1 = tf.Variable(tf.constant(0.0, shape=[128]))
    d_w_2 = tf.transpose(e_w_2)
    d_b_2 = tf.Variable(tf.constant(0.0, shape=[256]))
    d_w_3 = tf.transpose(e_w_1)
    d_b_3 = tf.Variable(tf.constant(0.0, shape=[520]))

    layer_1 = tf.nn.tanh(tf.add(tf.matmul(X, e_w_1), e_b_1))
    layer_2 = tf.nn.tanh(tf.add(tf.matmul(layer_1, e_w_2), e_b_2))
    encoded = tf.nn.tanh(tf.add(tf.matmul(layer_2, e_w_3), e_b_3))

    layer_4 = tf.nn.tanh(tf.add(tf.matmul(encoded, d_w_1), d_b_1))
    layer_5 = tf.nn.tanh(tf.add(tf.matmul(layer_4, d_w_2), d_b_2))
    decoded = tf.nn.tanh(tf.add(tf.matmul(layer_5, d_w_3), d_b_3))
    encoder_params =  []
    encoder_params.append({
        'e_w':e_w_1,
        'e_b':e_b_1,
        'd_w':d_w_3,
        'd_b':d_b_3
    })
    encoder_params.append({
        'e_w': e_w_2,
        'e_b': e_b_2,
        'd_w': d_w_2,
        'd_b': d_b_2
    })
    encoder_params.append({
        'e_w': e_w_3,
        'e_b': e_b_3,
        'd_w': d_w_1,
        'd_b': d_b_1
    })
    return encoded,decoded,encoder_params


def dnn(X,n_output):
    # DNN
    w_1 = tf.Variable(tf.truncated_normal([64, 128], stddev=0.1))
    b_1 = tf.Variable(tf.constant(0.0, shape=[128]))
    w_2 = tf.Variable(tf.truncated_normal([128, 128], stddev=0.1))
    b_2 = tf.Variable(tf.constant(0.0, shape=[128]))
    w_3 = tf.Variable(tf.truncated_normal([128, n_output], stddev=0.1))
    b_3 = tf.Variable(tf.constant(0.0, shape=[n_output]))

    layer_7 = tf.nn.tanh(tf.add(tf.matmul(X, w_1), b_1))
    layer_8 = tf.nn.tanh(tf.add(tf.matmul(layer_7, w_2), b_2))
    out = tf.nn.tanh(tf.add(tf.matmul(layer_8, w_3), b_3))
    return  out



def load_train_test():
    train_data = pd.read_csv(_TRAIN, header=0)
    # print(data.dtypes)
    train_x = scale(np.asarray(train_data.ix[:, 0:520]))
    train_y = np.asarray(train_data['BUILDINGID'].map(str) + train_data['FLOOR'].map(str))
    train_y = np.asarray(pd.get_dummies(train_y))

    print(train_y.shape)
    test_data = pd.read_csv(_TEST)
    test_x = scale(np.asarray(test_data.ix[:, 0:520]))
    test_y = np.asarray(test_data['BUILDINGID'].map(str) + test_data['FLOOR'].map(str))
    test_y = np.asarray(pd.get_dummies(test_y))
    return train_x,train_y,test_x,test_y

def train_a_tied_encoder(X,train_x, training_epochs,batch_size,total_batches):
    enc, dec, enc_params = encoder_tw_nn(X)
    enc_params_vals = []
    encoder_cost = tf.reduce_mean(tf.pow(X - dec, 2))
    encoder_opt = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(encoder_cost)
    # with tf.Session() as s:
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(training_epochs):
            epoch_costs = np.empty(0)
            for b in range(total_batches):
                offset = b * batch_size
                batch_x = train_x[offset:offset + batch_size, :]
                _, c = sess.run([encoder_opt, encoder_cost], feed_dict={X: batch_x})
                epoch_costs = np.append(epoch_costs, c)
        print("Epoch ", epoch, '  Cost: ', np.mean(epoch_costs))

        for para in enc_params:
            tmp_dd = {}
            for k,v in para:
                tmp_dd[k]  = sess.run(v)
            enc_params_vals.append(tmp_dd)
            #if k == 'ew1':
            #    print("After train para :", k, sess.run(v))
            #enc_params_vals[k] = sess.run(v)
    return enc_params_vals


def gen_1level_autoencoder(X,fixed_params,in_size, enc_size,enc_w_n='e_w',enc_b_n='e_b',dec_w_n='d_w',dec_b_n='d_b'):
    enc_pas = []
    dec_pas = []
    for param in fixed_params:
        enc_pas.append((param['e_w'],param['e_b']))
        dec_pas.insert(0,(param['d_w'],param['d_b']))
    # --------------------- Encoder -------------------- #

    layer = None
    for enc_p in enc_pas:
        e_w = enc_p[0]
        e_b = enc_p[1]
        if layer is None:
            layer = tf.tanh(tf.add(tf.matmul(X,e_w),e_b))
        else:
            layer = tf.tanh(tf.add(tf.matmul(layer,e_w),e_b))

    e_w = tf.Variable(tf.truncated_normal([in_size, enc_size], stddev=0.1))
    e_b = tf.Variable(tf.constant(0.0, shape=[enc_size]))

    if layer is None:
        layer = tf.tanh(tf.add(tf.matmul(X, e_w), e_b))
    else:
        layer = tf.tanh(tf.add(tf.matmul(layer, e_w), e_b))

    enc = layer

    # --------------------- Dencoder -------------------- #
    d_w_here = tf.Variable(tf.truncated_normal([enc_size, in_size], stddev=0.1))
    d_b_here = tf.Variable(tf.constant(0.0, shape=[in_size]))

    layer = tf.tanh(tf.add(tf.matmul(layer,d_w_here),d_b_here))
    for dec_p in dec_pas:
        d_w = dec_p[0]
        d_b = dec_p[1]
        layer = tf.tanh(tf.add(tf.matmul(layer, d_w), d_b))

    dec = layer
    #encoded = tf.tanh(tf.add(tf.matmul(X,e_w_1),e_b_1))
    #out = tf.tanh(tf.add(tf.matmul(encoded,d_w_1),d_b_1))
    ret = {
        enc_w_n:e_w,
        enc_b_n:e_b,
        dec_w_n:d_w_here,
        dec_b_n:d_b_here
    }
    encoder_cost = tf.reduce_mean(tf.pow(X - dec, 2))
    encoder_opt = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(encoder_cost)
    return encoder_opt,encoder_cost,ret


def train_autoenc(opt,cost,X ,train_x, t_epochs, b_size, t_batches,enc_params):
    enc_params_vals = {}
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(t_epochs):
            epoch_costs = np.empty(0)
            for b in range(t_batches):
                offset = b * b_size
                batch_x = train_x[offset:offset + b_size, :]
                _, c = sess.run([opt, cost], feed_dict={X: batch_x})
                epoch_costs = np.append(epoch_costs, c)
        print("Epoch ", epoch, '  Cost: ', np.mean(epoch_costs))

        for k, v in enc_params.items():
            enc_params_vals[k] = sess.run(v)
    return enc_params_vals


def train_a_lbl_encoder(X,train_x,t_epochs,b_size,t_batches):
    opt,cost,enc_params = gen_1level_autoencoder(X,[],in_size=520, enc_size=256)
    params1 = train_autoenc(opt,cost,X,train_x,t_epochs,b_size,t_batches,enc_params)
    opt,cost,enc_params = gen_1level_autoencoder(X,[params1],in_size=256,enc_size=128)
    params2 = train_autoenc(opt,cost,X,train_x,t_epochs,b_size,t_batches,enc_params)
    opt, cost, enc_params = gen_1level_autoencoder(X, [params1,params2], in_size=128, enc_size=64)
    params3 = train_autoenc(opt,cost,X,train_x,t_epochs,b_size,t_batches,enc_params)

    return [params1,params2,params3]



_TIED = False
def train_a_encoder(X,train_x, training_epochs,batch_size,total_batches):

    if _TIED:
        enc_params_vals =  train_a_tied_encoder(X,train_x, training_epochs,batch_size,total_batches)
    else:
        enc_params_vals = train_a_lbl_encoder(X,train_x, training_epochs,batch_size,total_batches)
    return enc_params_vals


def enc_dnn(name):
    print("Enter train tied weight enc")
    train_x,train_y,test_x,test_y = load_train_test()
    n_output = train_y.shape[1]
    X = tf.placeholder(tf.float32, shape=[None, 520])
    Y = tf.placeholder(tf.float32, [None, n_output])

    training_epochs = 20
    batch_size = 10
    total_batches = int(train_x.shape[0] / batch_size) + 1
#<<<<<<< HEAD
    #with tf.Session() as s:
    #with tf.Session() as sess:
    #    sess.run(tf.global_variables_initializer())
    #    for k,v in enc_params.items():
    #        if k == 'ew3':
    #            print("Before train para :", k,sess.run(v))
    #    for epoch in range(training_epochs):
    #        epoch_costs = np.empty(0)
    #        for b in range(total_batches):
    #            offset = b*batch_size
    #            batch_x = train_x[offset:offset+batch_size,:]
    #            _,c = sess.run([encoder_opt,encoder_cost],feed_dict={X:batch_x})
    #            epoch_costs = np.append(epoch_costs,c)
    #    print("Epoch ",epoch,'  Cost: ',np.mean(epoch_costs))#

    #    for k,v in enc_params.items():
    #        if k == 'ew3':
    #            print("After train para :", k,sess.run(v))
    #        enc_params_vals[k] = sess.run(v)
#=======
#    enc_params_vals = train_a_encoder(X,train_x,training_epochs,batch_size,total_batches)
#>>>>>>> f8f4e226908f6ba3825bac49d253dc388a5a7ec6

    #output = dnn_with_enc(enc_params_vals,X,n_output)
    #out_cost = tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits=output)
    #out_opt = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(out_cost)
    #correct = tf.equal(tf.argmax(Y,1),tf.argmax(output,1))
    #accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))
    #with tf.Session() as sess:
    #    sess.run(tf.global_variables_initializer())
    #    for epoch in range(training_epochs):
    #        epoch_costs = np.empty(0)
    #        for b in range(total_batches):
    #            offset = b*batch_size
    #            batch_x = train_x[offset:offset+batch_size,:]
    #            batch_y = train_y[offset:offset+batch_size,:]
    #            _,c = sess.run([out_opt,out_cost],feed_dict={X:batch_x,Y:batch_y})
    #            epoch_costs = np.append(epoch_costs,c)
    #        accuracy_train = sess.run(accuracy, feed_dict={X: train_x, Y: train_y})
    #        accuracy_test = sess.run(accuracy, feed_dict={X: test_x, Y: test_y})
    #        print("Epoch",epoch," cost: ",np.mean(epoch_costs)," accuracy train:",accuracy_train,
    #              " accuracy test: ",accuracy_test)
            #print("w3 of encoder",enc_params_vals['ew3'])




def dnn_with_enc(enc_pramas,X,n_output):

    layer = None
    for para in enc_pramas:
        e_w = tf.constant(para['e_w'])
        e_b = tf.constant(para['e_b'])

        if layer is None:
            layer = tf.tanh(tf.add(tf.matmul(X,e_w),e_b))
        else:
            layer = tf.tanh(tf.add(tf.matmul(layer,e_w),e_b))
    #e_w_1 = tf.constant(enc_pramas['ew1'])
    #e_b_1 = tf.constant(enc_pramas['eb1'])
    #e_w_2 = tf.constant(enc_pramas['ew2'])
    #e_b_2 = tf.constant(enc_pramas['eb2'])
    #e_w_3 = tf.constant(enc_pramas['ew3'])
    #e_b_3 = tf.constant(enc_pramas['eb3'])

    #layer1 = tf.tanh(tf.add(tf.matmul(X,e_w_1),e_b_1))
    #layer2 = tf.tanh(tf.add(tf.matmul(layer1,e_w_2),e_b_2))
    #encoded = tf.tanh(tf.add(tf.matmul(layer2,e_w_3),e_b_3))
    encoded = layer

    # now the encoded as input vector.
    w1 = tf.Variable(tf.truncated_normal([64,128],stddev=0.1))
    b1 = tf.Variable(tf.constant(0.0,shape=[128]))
    w2 = tf.Variable(tf.truncated_normal([128,128],stddev=0.1))
    b2 = tf.Variable(tf.constant(0.0,shape=[128]))
    w3 = tf.Variable(tf.truncated_normal([128,n_output],stddev=0.1))
    b3 = tf.Variable(tf.constant(0.0,shape=[n_output]))

    layer3 = tf.tanh(tf.add(tf.matmul(encoded,w1),b1))
    layer4 = tf.tanh(tf.add(tf.matmul(layer3,w2),b2))
    output = tf.tanh(tf.add(tf.matmul(layer4,w3),b3))
    return output


def nn_deprecated(X,output):
    # --------------------- Encoder -------------------- #
    e_w_1 = tf.Variable(tf.truncated_normal([520, 256], stddev=0.1))
    e_b_1 = tf.Variable(tf.constant(0.0, shape=[256]))
    e_w_2 = tf.Variable(tf.truncated_normal([256, 128], stddev=0.1))
    e_b_2 = tf.Variable(tf.constant(0.0, shape=[128]))
    e_w_3 = tf.Variable(tf.truncated_normal([128, 64], stddev=0.1))
    e_b_3 = tf.Variable(tf.constant(0.0, shape=[64]))
    # --------------------- Decoder  ------------------- #
    d_w_1 = tf.Variable(tf.truncated_normal([64, 128], stddev=0.1))
    d_b_1 = tf.Variable(tf.constant(0.0, shape=[128]))
    d_w_2 = tf.Variable(tf.truncated_normal([128, 256], stddev=0.1))
    d_b_2 = tf.Variable(tf.constant(0.0, shape=[256]))
    d_w_3 = tf.Variable(tf.truncated_normal([256, 520], stddev=0.1))
    d_b_3 = tf.Variable(tf.constant(0.0, shape=[520]))
    # --------------------- DNN  ------------------- #
    w_1 = tf.Variable(tf.truncated_normal([64, 128], stddev=0.1))
    b_1 = tf.Variable(tf.constant(0.0, shape=[128]))
    w_2 = tf.Variable(tf.truncated_normal([128, 128], stddev=0.1))
    b_2 = tf.Variable(tf.constant(0.0, shape=[128]))
    w_3 = tf.Variable(tf.truncated_normal([128, output], stddev=0.1))
    b_3 = tf.Variable(tf.constant(0.0, shape=[output]))
    #########################################################
    layer_1 = tf.nn.tanh(tf.add(tf.matmul(X, e_w_1), e_b_1))
    layer_2 = tf.nn.tanh(tf.add(tf.matmul(layer_1, e_w_2), e_b_2))
    encoded = tf.nn.tanh(tf.add(tf.matmul(layer_2, e_w_3), e_b_3))
    layer_4 = tf.nn.tanh(tf.add(tf.matmul(encoded, d_w_1), d_b_1))
    layer_5 = tf.nn.tanh(tf.add(tf.matmul(layer_4, d_w_2), d_b_2))
    decoded = tf.nn.tanh(tf.add(tf.matmul(layer_5, d_w_3), d_b_3))
    layer_7 = tf.nn.tanh(tf.add(tf.matmul(encoded, w_1), b_1))
    layer_8 = tf.nn.tanh(tf.add(tf.matmul(layer_7, w_2), b_2))
    out = tf.nn.softmax(tf.add(tf.matmul(layer_8, w_3), b_3))
    return (decoded, out)


def train_enc_dec_dnn_deprecated():
    train_x,train_y,test_x,test_y = load_train_test()
    # output_dims
    n_output = train_y.shape[1]
    X = tf.placeholder(tf.float32, shape=[None,520])
    Y = tf.placeholder(tf.float32,[None,n_output])

    dec, p_output = nn_deprecated(X,n_output)
    # mse

    encoder_cost = tf.reduce_mean(tf.pow(X-dec,2))
    # cross entry logit.

    encoder_opt = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(encoder_cost)


    training_epochs = 20
    batch_size = 10
    total_batches = int(train_x.shape[0]/batch_size) + 1

    #out_cost = -tf.reduce_sum(Y*tf.log(p_output))
    out_cost = tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=p_output)
    out_opt = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(out_cost)
    correct = tf.equal(tf.argmax(p_output, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(training_epochs):
            epoch_costs = np.empty(0)
            for b in range(total_batches):
                offset = b*batch_size
                batch_x = train_x[offset:(offset+batch_size),:]
                _,c  =sess.run([encoder_opt,encoder_cost],feed_dict={X:batch_x})
                epoch_costs = np.append(epoch_costs,c)
            print("Epoch: ",epoch,"Loss: ",np.mean(epoch_costs))
        for epoch in range(training_epochs):
            epoch_costs = np.empty(0)
            for b in range(total_batches):
                offset = b*batch_size
                batch_x = train_x[offset:offset+batch_size,:]
                batch_y = train_y[offset:offset+batch_size,:]
                _,c = sess.run([out_opt,out_cost],feed_dict={X:batch_x,Y:batch_y})
                #c = sess.run(out_cost,feed_dict={X:batch_x,Y:batch_y})
                epoch_costs = np.append(epoch_costs,c)

            accuracy_train = sess.run(accuracy, feed_dict={X:train_x,Y:train_y})
            accuracy_test = sess.run(accuracy,feed_dict={X:test_x,Y:test_y})

            print("Epoch: ",epoch," Loss: ",np.mean(epoch_costs)," Accuracy ",accuracy_train,
                  "Accuracy test:",accuracy_test)
            print("encode end w:")



def main():
    enc_dnn('tied_weight')
    #train_enc_dec_dnn()


if __name__ == "__main__":
    main()