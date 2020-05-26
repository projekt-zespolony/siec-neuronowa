import tensorflow as tf


def main():
    data = [
        [0, 25, 999, 50, 20],
        [0, 20, 995, 55, 25],
        [0, 10, 990, 45, 30],
        [0, 30, 990, 45, 30],
        [0, 30, 1000, 45, 30],
        [0, 30, 1000, 40, 30],
        [0, 30, 1000, 40, 40],
        [0, 30, 1000, 40, 20],
        [0, 30, 1005, 40, 20],
        [0, 30, 1005, 50, 20],
        [0, 30, 1005, 30, 20],
        [0, 30, 1000, 30, 20],
        [0, 30, 995, 30, 20],
        [0, 30, 990, 30, 20],
        [1590229879, 24.56, 1003.72, 46.86, 16.63], # 0 1
        [1590249652, 24.83, 1001.85, 50.01, 53.24] # 1 1
    ]

    with tf.Session(graph = tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, ['tag'], 'model')
        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("input_input:0")
        model = graph.get_tensor_by_name("output/Sigmoid:0")
        y = sess.run(model, {x: data})

        for n in range(len(y)):
            print(data[n], y[n])

if __name__ == '__main__':
    main()
