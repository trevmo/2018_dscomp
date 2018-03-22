"""
This file contains the definition of a Tensorflow-based neural network. It is
designed for detecting numeric digits or simple mathematical symbols from an
input image.

@author trevmo
"""

import tensorflow as tf

def form_model(features, labels, mode):
	"""
	"""
	# shape = (24,24)
	#input_layer = tf.layers.Input(shape=[-1, 24, 24, 1], name="input")
	input_layer = tf.reshape(features["x"], [-1, 24, 24, 1])
	# shape = (24, 24, 32)
	first_conv = tf.layers.conv2d(inputs=input_layer,
									filters=32,
									kernel_size=(4, 4),
									activation=tf.nn.relu,
									padding="same",
									name="first_conv")
	# shape = (12, 12, 32)
	first_pool = tf.layers.max_pooling2d(inputs=first_conv,
											pool_size=(2, 2),
											strides=2,
											name="first_pool")
	# shape = (12, 12, 64)
	second_conv = tf.layers.conv2d(inputs=first_pool,
									filters=64,
									kernel_size=(4, 4),
									activation=tf.nn.relu,
									padding="same",
									name="second_conv")
	# shape = (6, 6, 64)
	second_pool = tf.layers.max_pooling2d(inputs=second_conv,
											pool_size=(2, 2),
											strides=2,
											name="second_pool")
	# shape = (6 * 6 * 64)
	second_pool_flat = tf.reshape(second_pool, [-1, 6 * 6 * 64])
	print(second_pool_flat.get_shape())
	# shape = (1024)
	dense_layer = tf.layers.dense(inputs=second_pool_flat,
									units=1024,
									activation=tf.nn.relu,
									name="dense")
	dropout_layer = tf.layers.dropout(inputs=dense_layer,
										rate=0.5,
										training=mode == tf.estimator.ModeKeys.TRAIN)
	# shape = (13)
	output_layer = tf.layers.dense(inputs=dropout_layer,
										units=13,
										name="outputs")
	results = {
		"classes": tf.argmax(input=output_layer, axis=1),
		"probabilities": tf.nn.softmax(output_layer, name="soft_max_prob")
	}
	loss = tf.losses.sparse_softmax_cross_entropy(labels=labels,
													logits=output_layer)

	if mode == tf.estimator.ModeKeys.TRAIN:
		optimizer = tf.train.AdamOptimizer()
		train_optimizer = optimizer.minimize(loss=loss,
												global_step=tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(mode=mode,
											loss=loss,
											train_op=train_optimizer)
	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode=mode, predictions=results)
