import os
import numpy as np
import tensorflow as tf


def _tensor_to_bytes_feature(value):
    """Converts the tensor to bytes."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def create_tfrecords_with_writer(images, labels, writer):
    """Creates tfrecords for given images. """
    for image, label in zip(images, labels):
        image_raw = image.astype("float32").tostring()
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "image_raw": _tensor_to_bytes_feature(image_raw),
                    "label": _int64_feature(int(label)),
                }
            )
        )
        writer.write(example.SerializeToString())


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    images_train = np.load("binary.train.npy")
    labels_train = np.load("label.train.npy")
    images_test = np.load("binary.test.npy")
    labels_test = np.load("label.test.npy")

    with tf.python_io.TFRecordWriter("train") as writer:
        create_tfrecords_with_writer(images_train, labels_train, writer)

    with tf.python_io.TFRecordWriter("test") as writer:
        create_tfrecords_with_writer(images_test, labels_test, writer)
