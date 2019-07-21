import tensorflow as tf

def model_fn(name):
    if name == "lenet5":
        return lenet5
    if name == "larger":
        return lenet_more_layer

def lenet5(features, labels, mode, params):
    net = tf.reshape(features["image"], [-1, 28, 28, 1])
    net = tf.layers.conv2d(
        inputs=net,
        filters=6,
        kernel_size=[5, 5],
        strides=(1, 1),
        padding='valid',
        activation=tf.nn.relu,
        use_bias=True)
    net = tf.layers.max_pooling2d(
        inputs=net,
        pool_size=[2, 2],
        strides=2)
    net = tf.layers.conv2d(
        inputs=net,
        filters=16,
        kernel_size=[5, 5],
        strides=(1, 1),
        padding='valid',
        activation=tf.nn.relu,
        use_bias=True)
    net = tf.layers.max_pooling2d(
        inputs=net,
        pool_size=[2, 2],
        strides=2)
    net = tf.layers.dense(
        inputs=tf.reshape(net, [-1, 4 * 4 * 16]),
        units=120,
        activation=tf.nn.relu)
    net = tf.layers.dense(
        inputs=net,
        units=84,
        activation=tf.nn.relu)

    # Logits Layer
    logits = tf.layers.dense(inputs=net, units=10)

    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits)
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def lenet_more_layer(features, labels, mode, params):
    net = tf.reshape(features["image"], [-1, 28, 28, 1])
    net = tf.layers.conv2d(
        inputs=net,
        filters=32,
        kernel_size=[5, 5],
        strides=(1, 1),
        padding='valid',
        activation=tf.nn.relu,
        use_bias=True)
    net = tf.layers.max_pooling2d(
        inputs=net,
        pool_size=[2, 2],
        strides=2)
    net = tf.layers.conv2d(
        inputs=net,
        filters=64,
        kernel_size=[5, 5],
        strides=(1, 1),
        padding='valid',
        activation=tf.nn.relu,
        use_bias=True)
    net = tf.layers.max_pooling2d(
        inputs=net,
        pool_size=[2, 2],
        strides=2)
    net = tf.layers.dense(
        inputs=tf.reshape(net, [-1, 4 * 4 * 64]),
        units=1024,
        activation=tf.nn.relu)
    net = tf.layers.dense(
        inputs=net,
        units=64,
        activation=tf.nn.relu)

    # Logits Layer
    logits = tf.layers.dense(inputs=net, units=10)

    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits)
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
