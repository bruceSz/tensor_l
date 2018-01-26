
import tensorflow as tf
import numpy as np

def main():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("../data",one_hot=True)
    #print(type(mnist))

    chunk_size = 28
    chunk_n = 28
    rnn_size = 256

    n_output_layer = 10

    batch_size = 100
    X = tf.placeholder('float',[None,chunk_n,chunk_size])
    Y = tf.placeholder('float')
    predict = recurrent_neural_network(X,chunk_size,chunk_n,rnn_size,n_output_layer)
    cost_f = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(predict,Y))
    optimizer = tf.train.AdamOptimizer().minimize(cost_f)
    epochs = 13
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        epoch_loss = 0
        for epoch in range(0,epochs):
            for i in range(int(mnist.train.num_examples/batch_size)):
                x, y = mnist.train.next_batch(batch_size)
                x = x.reshape([batch_size,chunk_n,chunk_size])
                _,c  = sess.run([optimizer,cost_f],feed_dict={X:x,Y:y})
                epoch_loss += c
            print(epoch,":",epoch_loss)
        correct = tf.equal(tf.argmax(predict,1),tf.argmax(Y,1))
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        print("Accuracy:",accuracy.eval({X:mnist.test.images.reshape(-1,chunk_n,chunk_size),
                                            Y:mnist.test.labels}))



def recurrent_neural_network(data,chunk_size,chunk_n,rnn_size,n_output_layers):
    layer = {'w_':tf.Variable(tf.random_normal([rnn_size,n_output_layers])),
             'b_':tf.Variable(tf.random_normal([n_output_layers]))}
    # TODO. seems tensorflow 1.1.0 has no rnn cell?
    lstm_cell = tf.nn.rnn_cell_impl._RNNCell(rnn_size)
    data = tf.transpose(data)
    data = tf.reshape(data,[-1,chunk_size])
    data = tf.split(0,chunk_n,data)
    outputs,status = tf.nn.dynamic_rnn(lstm_cell,data,dtype=tf.float32)
    output = tf.add(tf.matmul(outputs[-1],layer['w_']),layer['b_'])
    return output




if __name__ == "__main__":
    main()