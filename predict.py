"""Trains a neural network for MNIST."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf
import models
import mnist_input
import numpy as np


def main(_):
    estimator = tf.estimator.Estimator(
        model_fn=models.model_fn(FLAGS.model), model_dir=FLAGS.model)

    predictions = estimator.predict(input_fn=mnist_input.eval_input_fn)
    pred = [p["classes"] for p in predictions]
    labels = np.load("label.test.npy")

    print("Accuracy is {:.4f}".format(sum(pred == labels) / len(labels)))
    np.save("{:s}.npy".format(FLAGS.model), pred)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="lenet5",
        help="The gpu used for training.")
    parser.add_argument(
        "--gpu",
        type=str,
        default="1",
        help="The gpu used for training.")
    FLAGS, unparsed=parser.parse_known_args()

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=FLAGS.gpu
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
