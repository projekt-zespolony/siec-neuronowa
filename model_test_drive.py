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
        [0, 30, 990, 30, 20]
    ]

    with tf.Session(graph = tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, ['tag'], 'model')
        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("input_input:0")
        model = graph.get_tensor_by_name("output/Sigmoid:0")
        print(sess.run(model, {x: data}))

if __name__ == '__main__':
    main()
