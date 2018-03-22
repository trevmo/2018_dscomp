"""
This file contains functions to assist with handling and manipulating data of
various kinds. It is designed to assist with taking input data from csv's,
reformatting for use in a neural network, and saving results away in files.

@author trevmo
"""

import numpy as np 
import pandas as pd

def read_data(path, filename, drop_col="index"):
	"""
	Read file data in as a dataframe, then drop the specified column and convert
	to a numpy array.

	Inputs:
	- path: path to the file
	- filename: name of the data file
	- drop_col: label of the column to drop

	Return:
	- numpy array of the data
	"""
	data = pd.read_csv(path + filename, sep=",")
	data = data.drop(drop_col, axis=1)
	return data.as_matrix()


def output_data(data, outfile):
	"""
	TODO
	"""
    pass


def format_data(arr, dimen, data_type):
	"""
	Format the data for use in a neural network. 
	
	For both the training and test data, one row in the numpy array represents
	a single image. With the test data, each image is five images of the same
	dimensions as the training data stitched together.
	- If it is training data, then, per row, reshape it into a numpy array of
	the specified dimensions.
	- If it is test data, then, per row, reshape into the correct dimensions and
	split into arrays of the same dimensions as the training data.

	Inputs:
	- arr: numpy array of data
	- dimen: tuple of dimensions
	- data_type: {"train", "test"}

	Return:
	- for data of type:
		- "train": list of numpy arrays of specified dimensions
		- "test": list of lists of numpy arrays of (rows, cols/rows)
	"""
	if type(dimen) != tuple:
		print("Error: expected tuple type.")
		return None
	return {
		"train": [np.reshape(row, dimen) for row in arr],
		"test": [np.hsplit(np.reshape(row, dimen), int(dimen[1] / dimen[0])) for row in arr]
	}.get(data_type)
