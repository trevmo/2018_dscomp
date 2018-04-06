"""
This file contains methods for forming a TFLearn-based CNN. It is
designed for detecting numeric digits or simple mathematical symbols from an
input image.

Initially I implemented this with vanilla Tensorflow and utilized this reference
(https://www.tensorflow.org/tutorials/layers) to help understand how to construct
a neural network model in Tensorflow.

However, after realizing that it would be helpful to include data preprocessing
and augmentation, I found that TFLearn provided those as built-in class/methods
to their wrapper of Tensorflow. I then converted my Tensorflow model into a
TFLearn model.

@author trevmo
"""
from __future__ import division, print_function, absolute_import
import tflearn
import tflearn.layers as tfl


def form_model(params):
	"""
	Create a TFLearn model with the following characteristics:
	- data preprocessing (std. normalization)
	- data augmentation (rotation of up to 15 degrees)
	- two convolutional and pooling layers
	- batch normalization
	- utilizes Adam optimizer for training
	- relies on categorical cross entropy loss

	Inputs:
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
	- model for input into the DNN class constructor
	"""
	image_processor = tflearn.ImagePreprocessing()
	image_processor.add_featurewise_stdnorm()

	image_augmentator = tflearn.ImageAugmentation()
	image_augmentator.add_random_rotation(max_angle=15.0)

	input_layer = tfl.input_data(shape=params["input_shape"],
		data_preprocessing=image_processor,
		data_augmentation=image_augmentator)
	first_conv = tfl.conv_2d(input_layer,
		params["conv_filters"],
		params["conv_kernel"],
		activation='relu',
		padding='same')
	first_pool = tfl.max_pool_2d(first_conv,
		params["pool_size"],
		strides=params["pool_strides"])
	first_pool_norm = tfl.batch_normalization(first_pool)

	second_conv = tfl.conv_2d(first_pool_norm,
		2 * params["conv_filters"],
		params["conv_kernel"],
		activation='relu',
		padding='same')
	second_pool = tfl.max_pool_2d(second_conv,
		params["pool_size"],
		params["pool_strides"])
	second_pool_norm = tfl.batch_normalization(second_pool)

	dimensions = second_pool.get_shape().as_list()
	second_pool_flat = tfl.fully_connected(second_pool_norm,
		dimensions[1],
		activation='relu')
	flat_dropout = tfl.dropout(second_pool_flat, params["dropout_rate"])

	dense_layer = tfl.fully_connected(flat_dropout, 
		params["dense_units"],
		activation='relu')
	dense_layer_dropout = tfl.dropout(dense_layer, params["dropout_rate"])

	output_layer = tfl.fully_connected(dense_layer_dropout,
		params["output_units"],
		activation='softmax')

	return tfl.regression(output_layer,
		optimizer='adam',
		loss='categorical_crossentropy',
		name='results')

def generate_classifier(model, model_dir):
	"""
	Form a Deep Neural Network ("classifier") from the model.

	Input:
	- model: TFLearn network/model
	- model_dir: path to store checkpoints

	Return:
	- CNN classifier
	"""
	return tflearn.DNN(model, checkpoint_path=model_dir)


def train_classifier(classifier, data, percent_val, num_epochs, num_steps):
	"""
	Train the classifier using the given data and parameters.

	Inputs:
	- classifier: tflearn.DNN()
	- data: dictionary with the following elements:
		- inputs: numpy array of training data
		- labels: numpy array of training labels
	- percent_val: percent of training data to use for validation
	- num_epochs: number of epochs to run training
	- num_steps: number of steps to store snapshots at

	Return:
	- trained classifier
	"""
	classifier.fit(data["inputs"],
		data["labels"],
		n_epoch=num_epochs,
		validation_set=percent_val,
		snapshot_step=num_steps,
		show_metric=True,
		run_id="cnn_classifier")
	return classifier


def predict_with_classifier(data, classifier):
	"""
	Predict the classes/labels for the given data.

	Inputs:
	- data: input data to predict labels for
	- classifier: trained CNN from model

	Return:
	- list of results
	"""
	return classifier.predict_label(data)
