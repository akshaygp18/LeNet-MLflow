import tensorflow as tf

def load_mnist_data():
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    return train_images, train_labels, test_images, test_labels
