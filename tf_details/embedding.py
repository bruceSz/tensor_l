# -*- coding=utf-8 -*-

import tensorflow as tf
import numpy as np

a = [[0.1, 0.2, 0.3], [1.1, 1.2, 1.3], [2.1, 2.2, 2.3], [3.1, 3.2, 3.3], [4.1, 4.2, 4.3]]
a = np.asanyarray(a)

idx1 = tf.Variable([0,2,3,1],tf.int32)
idx2 = tf.Variable([[0,2,3,1],[4,0,2,2]],tf.int32)
out1 = tf.nn.embedding_lookup(a,idx1)
out2 = tf.nn.embedding_lookup(a,idx2)

init = tf.global_variables_initializer()
print("="*10)
with tf.Session() as s:
    s.run(init)
    print(s.run(out1))
    print out1
    print "="*10
    print(s.run(out2))
    print(out2)


print("="*10)

param1 = tf.constant([1,2])
#param1 = np.asanyarray([1,2])
#param2 = np.asanyarray([10,20])
param2 = tf.constant([10,20])

ids = tf.constant([2,0,2,1,2,3],tf.int32)
ret = tf.nn.embedding_lookup([param1,param2],ids)
#print([param1,param2])
#init = tf.global_variables_initializer()
with tf.Session() as s:
    s.run(init)
    print(s.run([param1,param2]))
    print(s.run(ret))
    print(ret)

