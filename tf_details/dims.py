
import  tensorflow as tf
a = tf.constant([[1,2,3],[4,5,6]])
b = tf.expand_dims(a,-1)
with tf.Session() as s:
    print(s.run(a))
    print(s.run(b))