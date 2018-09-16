
import tensorflow as tf


def complex_graph():
    g = tf.Graph()
    with g.as_default():
        sess = tf.Session()
        w = tf.Variable(tf.zeros([10,5]))
        v = tf.placeholder(tf.float32,[None,10])
        result = tf.matmul(v,w)
    print(g.as_graph_def())

def simple_g_def():
    w = tf.Variable(tf.random_uniform([784,100],-1,1))
    print(tf.get_default_graph().as_graph_def())

if __name__ == "__main__":
    simple_g_def()