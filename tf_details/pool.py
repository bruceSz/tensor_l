import tensorflow as tf

def max_pool():

    a =  tf.constant([
        [
            [1.0,2.0,3.0,4.0],
            [5.0,6.0,7.0,8.0],
            [i for i in range(8,4,-1)],
            [i for i in range(4,0,-1)]
        ],
        [
            [i for i in range(4,0,-1)],
            [i for i in range(8,4,-1)],
            [i for i in range(1,5)],
            [5.0, 6.0, 7.0, 8.0]
        ],
    ])
    print(a)
    with tf.Session() as sess:
        print("image: ")
        print(sess.run(a))

    a = tf.reshape(a,[1,4,4,2])
    print(a)
    pooling = tf.nn.max_pool(a,[1,2,2,1],[1,1,1,1],padding='VALID')

    with tf.Session() as sess:
        print("Image: (after reshape)")
        print(sess.run(a))
        print("After pooling: ")
        ret = sess.run(pooling)
        print(ret)


def simple_example():
    #a = tf.constant([[[1,2]],[[3,4]]])
    a = tf.constant([
        [
            [i for i in range(1,5)],
            [i for i in range(5,9) ],
            [i for i in range(8, 4, -1)],
            [i for i in range(4,0,-1)]
        ],
        [
            [i for i in range(4,0,-1)],
            [i for i in range(8,4,-1)],
            [i for i in range(1, 5)],
            [i for i in range(5, 9)]
        ],
    ])
    print(a)


def main():
    #simple_example()
    max_pool()

if __name__ == "__main__":
    main()