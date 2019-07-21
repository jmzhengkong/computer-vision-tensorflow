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


def main(_):
    estimator = tf.estimator.Estimator(
        model_fn=models.model_fn(FLAGS.model), model_dir=FLAGS.model)

    train_spec = tf.estimator.TrainSpec(
        input_fn=mnist_input.train_input_fn, max_steps=FLAGS.max_steps)

    eval_spec = tf.estimator.EvalSpec(
        input_fn=mnist_input.eval_input_fn, throttle_secs=10)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
    parser.add_argument(
        "--max_steps",
        type=int,
        default=100000,
        help="The maximum training steps.")
    FLAGS, unparsed = parser.parse_known_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
