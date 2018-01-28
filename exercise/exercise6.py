import numpy as np
import tensorflow as tf

import pandas as pd
from sklearn.preprocessing import scale

_TRAIN = "../data/UJIndoorLoc/trainingData.csv"
_TEST = "../data/UJIndoorLoc/validationData.csv"

def encoder_dnn(X,n_output):
    # encoder
    e_w_1 = tf.Variable(tf.truncated_normal([520,256],stddev=0.1))
    e_b_1 = tf.Variable(tf.constant(0.0,shape=[256]))
    e_w_2 = tf.Variable(tf.truncated_normal([256,128],stddev=0.1))
    e_b_2 = tf.Variable(tf.constant(0.0,shape=[128]))
    e_w_3 = tf.Variable(tf.truncated_normal([128,64],stddev=0.1))
    e_b_3 = tf.Variable(tf.constant(0.0,shape=[64]))
    # decoder
    #d_w_1 = tf.Variable(tf.truncated_normal([64,128],stddev=-0.1))
    d_w_1 = tf.transpose(e_w_3)
    d_b_1 = tf.Variable(tf.constant(0.0,shape=[128]))
    #d_w_2 = tf.Variable(tf.truncated_normal([128, 256], stddev=-0.1))
    d_w_2 = tf.transpose(e_w_2)
    d_b_2 = tf.Variable(tf.constant(0.0, shape=[256]))
    #d_w_3 = tf.Variable(tf.truncated_normal([256, 520], stddev=-0.1))
    d_w_3 = tf.transpose(e_w_1)
    d_b_3 = tf.Variable(tf.constant(0.0, shape=[520]))


    # DNN
    w_1 = tf.Variable(tf.truncated_normal([64,128],stddev=0.1))
    b_1 = tf.Variable(tf.constant(0.0,shape=[128]))
    w_2 = tf.Variable(tf.truncated_normal([128, 128], stddev=0.1))
    b_2 = tf.Variable(tf.constant(0.0, shape=[128]))
    w_3 = tf.Variable(tf.truncated_normal([128, n_output], stddev=0.1))
    b_3 = tf.Variable(tf.constant(0.0, shape=[n_output]))

    layer_1 = tf.nn.tanh(tf.add(tf.matmul(X,e_w_1),e_b_1))
    layer_2 = tf.nn.tanh(tf.add(tf.matmul(layer_1,e_w_2),e_b_2))
    encoded = tf.nn.tanh(tf.add(tf.matmul(layer_2,e_w_3),e_b_3))

    layer_4 = tf.nn.tanh(tf.add(tf.matmul(encoded,d_w_1),d_b_1))
    layer_5 = tf.nn.tanh(tf.add(tf.matmul(layer_4,d_w_2),d_b_2))
    decoded = tf.nn.tanh(tf.add(tf.matmul(layer_5,d_w_3),d_b_3))

    layer_7 = tf.nn.tanh(tf.add(tf.matmul(encoded,w_1),b_1))
    layer_8 = tf.nn.tanh(tf.add(tf.matmul(layer_7,w_2),b_2))
    out = tf.nn.tanh(tf.add(tf.matmul(layer_8,w_3),b_3))
    return (decoded, out)


def train_enc_dec_dnn():
    train_data = pd.read_csv(_TRAIN,header=0)
    #print(data.dtypes)
    train_x = scale(np.asarray(train_data.ix[:,0:520]))
    train_y = np.asarray(train_data['BUILDINGID'].map(str)+train_data['FLOOR'].map(str))
    train_y = np.asarray(pd.get_dummies(train_y))

    print(train_y.shape)
    test_data = pd.read_csv(_TEST)
    test_x = scale(np.asarray(test_data.ix[:,0:520]))
    test_y = np.asarray(test_data['BUILDINGID'].map(str)+test_data['FLOOR'].map(str))
    test_y = np.asarray(pd.get_dummies(test_y))
    print(test_y.shape)
    # output_dims
    output = train_y.shape[1]
    X = tf.placeholder(tf.float32, shape=[None,520])
    Y = tf.placeholder(tf.float32,[None,output])
    dec, p_output = encoder_dnn(X,output)
    # mse

    encoder_cost = tf.reduce_mean(tf.pow(X-dec,2))
    # cross entry logit.
    #out_cost = -tf.reduce_sum(Y*tf.log(p_output))
    out_cost = tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits=p_output)
    encoder_opt = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(encoder_cost)
    out_opt = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(out_cost)

    correct = tf.equal(tf.argmax(p_output,1),tf.argmax(Y,1))
    accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))

    training_epochs = 20
    batch_size = 10
    total_batches = int(train_data.shape[0]/batch_size) + 1
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




def main():
    train_enc_dec_dnn()


if __name__ == "__main__":
    main()