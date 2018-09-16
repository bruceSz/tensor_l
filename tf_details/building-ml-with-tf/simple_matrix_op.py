
def main():
    import tensorflow as tf
    sess  = tf.InteractiveSession()
    x = tf.constant([[1.0,2.0],[3.0,4.0]])
    print(tf.transpose(x).eval())
    y = tf.constant([[3.0, 4.0, 5.0], [4.0, 5.0, 6.0]])
    print(y.eval())
    print(tf.matmul(x,y).eval())
    print(tf.matrix_determinant(x).eval())
    print(tf.matrix_inverse(x).eval())
    print(tf.matrix_solve(x,[[1],[1]]).eval())

def redu():
    import tensorflow as tf
    with tf.Session() as sess:
        x = tf.constant([
                [[1,1],[2,2],[3,3]],
                [[3,3],[2,2],[-1,-1]],
                [[-1,-1],[-2,-2],[-3,-3]]
        ])
        print(tf.reduce_prod(x,reduction_indices=0).eval())

if __name__ == "__main__":
    redu()