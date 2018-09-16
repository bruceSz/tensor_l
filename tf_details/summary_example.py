
import tensorflow as tf
import numpy as np

def gen_data():
    x_data = np.float32(np.random.rand(2,100))
    y = np.dot([0.1,0.2],x_data)+0.3
    return x_data,y

def linear_reg(n_samples,dim):
    x_ = tf.placeholder(tf.float32,shape=(dim,None))
    y  = tf.placeholder(tf.float32)
    b = tf.Variable(tf.zeros([1]),name='b')
    w = tf.Variable(tf.random_normal([1,dim],-1,1),name='w')
    print(w.shape)
    y_pred = tf.add(tf.matmul(w, x_,name="matmul"),b,name='add')
    loss = tf.reduce_mean(tf.square(tf.subtract(y,y_pred,name="sub"),name="square"),name="reduce_mean")
    opt = tf.train.GradientDescentOptimizer(0.001,name="optimizer")
    train = opt.minimize(loss)
    sum = [tf.summary.histogram("w",w),tf.summary.histogram("b",b),tf.summary.scalar('loss',loss)]
    print(tf.__version__)
    #sum_op = tf.summary.(sum)
    return train

def main():
    train_op, sum_op = linear_reg(100,2)
    x,y = gen_data()
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables)


if __name__ == "__main__":
    main()