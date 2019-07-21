import tensorflow as tf
import os


def _parse_function(example_proto):
    """Parses raw bytes into tensors."""
    features = {
        "image_raw": tf.FixedLenFeature((), tf.string, default_value=""),
        "label": tf.FixedLenFeature((), tf.int64),
    }
    parsed_features = tf.parse_single_example(example_proto, features)
    output_features = {
        "image": tf.reshape(
            tf.decode_raw(parsed_features["image_raw"], tf.float32),
            [28, 28],
        ),
    }
    labels = tf.cast(parsed_features["label"], tf.int32)
    # Returns a tuple (features, labels)
    return output_features, labels


def train_input_fn():
    """An input function for training."""
    dataset = tf.data.TFRecordDataset("train")
    dataset = dataset.map(_parse_function)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(32)
    return dataset.make_one_shot_iterator().get_next()


def eval_input_fn():
    """An input function for evaluation."""
    dataset = tf.data.TFRecordDataset("test")
    dataset = dataset.map(_parse_function)
    dataset = dataset.batch(32)
    return dataset.make_one_shot_iterator().get_next()
