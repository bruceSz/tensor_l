
import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../data",one_hot=True)



def create_network_cnn(data,n_output_layer):
    weights = {'w_conv_1':tf.Variable(tf.random_normal([5,5,1,32])),
               'w_conv_2':tf.Variable(tf.random_normal([5,5,32,64])),
               'w_fc':tf.Variable(tf.random_normal([7*7*64,1024])),
               'out':tf.Variable(tf.random_normal([1024,n_output_layer]))}
    biases = {'b_conv_1':tf.Variable(tf.random_normal([32])),
              'b_conv_2':tf.Variable(tf.random_normal([64])),
              'b_fc':tf.Variable(tf.random_normal([1024])),
              'b_out':tf.Variable(tf.random_normal([n_output_layer]))}
    data = tf.reshape(data,[-1,28,28,1])
    conv1 = tf.nn.relu(tf.add(tf.nn.conv2d(data,weights['w_conv_1'],strides=[1,1,1,1],
                                           padding='SAME'),biases['b_conv_1']))
    conv1 = tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    conv2 = tf.nn.relu(tf.add(tf.nn.conv2d(conv1,weights['w_conv_2'],strides=[1,1,1,1],
                                           padding='SAME'),biases['b_conv_2']))
    conv2 = tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    fc = tf.reshape(conv2,[-1,7*7*64])
    fc = tf.nn.relu(tf.add(tf.matmul(fc,weights['w_fc']),biases['b_fc']))

    output = tf.add(tf.matmul(fc,weights['out']),biases['b_out'])
    return output


def main():
    n_output_layers = 10
    batch_size  = 10
    X = tf.placeholder('float',[None,28*28])
    Y = tf.placeholder('float')

    # train
    predict = create_network_cnn(X,n_output_layers)
    cost_f = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict,labels=Y))
    optimizer = tf.train.AdamOptimizer().minimize(cost_f)
    epochs = 10
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        epoch_loss = 0
        for epoch in range(epochs):
            for i in range(int(mnist.train.num_examples/batch_size)):
                x, y = mnist.train.next_batch(batch_size)
                _,c = sess.run([optimizer,cost_f],feed_dict={X:x,Y:y})
                epoch_loss += c
            print("Epoch: ",epoch,': loss:',epoch_loss)
        correct = tf.equal(tf.argmax(predict,1),tf.argmax(Y,1))
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        print("Accuracy: ",accuracy.eval({X:mnist.test.images,Y:mnist.test.labels}))



if __name__ == "__main__":
    main()