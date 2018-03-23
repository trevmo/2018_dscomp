"""
This file contains methods for forming a Tensorflow-based CNN. It is
designed for detecting numeric digits or simple mathematical symbols from an
input image.

@author trevmo
"""

import tensorflow as tf


def form_model(features, labels, mode, params):
	"""
	Create a Tensorflow model with the following characteristics:
	- input layer -> convolutional -> max pooling -> convolutional -> max pooling
	-> flattened pooling -> dense -> (dropout) -> output layer
	- utilizes Adam optimizer for training
	- relies on sparse softmax cross entropy loss function

	Inputs:
	- features: input features for the model
	- labels: input labels
	- mode: {TRAIN, EVALUATE, PREDICT}
	- params: dictionary containing the following elements:
		- input_shape: shape of the input tensor
		- conv_filters: number of filters to apply to the first convolution layer
			- second convolution layer is 2*this value
		- conv_kernel: shape of the convolutional kernel for both layers
		- pool_size: shape of the pool for both layers
		- pool_strides: size of the pool stride for both layers
		- dense_units: dimension of the dense layer prior to output
		- dropout_rate: rate of dropout in training
		- output_units: number of classes

	Return:
	- estimator for the given mode of use
	"""
	
	input_layer = tf.reshape(features["x"], params["input_shape"])
	
	first_conv = tf.layers.conv2d(inputs=input_layer,
		filters=params["conv_filters"],
		kernel_size=params["conv_kernel"],
		activation=tf.nn.relu,
		padding="same",
		name="first_conv")
	
	first_pool = tf.layers.max_pooling2d(inputs=first_conv,
		pool_size=params["pool_size"],
		strides=params["pool_strides"],
		name="first_pool")
	
	second_conv = tf.layers.conv2d(inputs=first_pool,
		filters=2 * params["conv_filters"],
		kernel_size=params["conv_kernel"],
		activation=tf.nn.relu,
		padding="same",
		name="second_conv")
	
	second_pool = tf.layers.max_pooling2d(inputs=second_conv,
		pool_size=params["pool_size"],
		strides=params["pool_strides"],
		name="second_pool")
	
	dimensions = second_pool.get_shape().as_list()
	second_pool_flat = tf.reshape(
		second_pool, [-1, (dimensions[1]**2) * dimensions[-1]])
	
	
	dense_layer = tf.layers.dense(inputs=second_pool_flat,
		units=params["dense_units"],
		activation=tf.nn.relu,
		name="dense")
	
	dropout_layer = tf.layers.dropout(inputs=dense_layer,
		rate=params["dropout_rate"],
		training=mode == tf.estimator.ModeKeys.TRAIN)
	
	output_layer = tf.layers.dense(inputs=dropout_layer,
		units=params["output_units"],
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


def train_model(data, batch_size, num_steps, model_params):
	"""
	Train the model using the given data and parameters.

	Inputs:
	- data: dictionary with the following elements:
		- train_data: numpy array of training data
		- train_labels: numpy array of training labels
	- batch_size: size of mini-batches for training
	- num_steps: number of steps to run the training
	- model_params: dictionary containing the following elements:
		- input_shape: shape of the input tensor
		- conv_filters: number of filters to apply to the first convolution layer
			- second convolution layer is 2*this value
		- conv_kernel: shape of the convolutional kernel for both layers
		- pool_size: shape of the pool for both layers
		- pool_strides: size of the pool stride for both layers
		- dense_units: dimension of the dense layer prior to output
		- dropout_rate: rate of dropout in training
		- output_units: number of classes
	"""
	tf.logging.set_verbosity(tf.logging.INFO)

	classifier = tf.estimator.Estimator(
		model_fn=form_model,
		params=model_params,
		model_dir="model")

	train_input_fn = tf.estimator.inputs.numpy_input_fn(
		x={"x": data["inputs"]},
		y=data["labels"],
		batch_size=batch_size,
		num_epochs=None,
		shuffle=True)
	classifier.train(
		input_fn=train_input_fn,
		steps=num_steps,
		hooks=[])

	tf.app.run()
