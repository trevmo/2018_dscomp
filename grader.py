"""
This file contains the definition of the Grader class. It is used in conjunction
with the CNN model from model.py.


@author trevmo
"""

import numpy as np
import model as mdl
import csv


class Grader(object):
	"""
	The Grader class is designed to analyze the accuracy of images of handwritten
	equations.

	These equations can take the form of:
	- a + b = c
	- c = a - b
	It reformats these images into one per numeric value or symbol and then hands
	them off to the classifier to predict the values. From there it checks the
	correctness of the equation and stores the results in a file.
	"""

	PLUS = 10
	MINUS = 11
	EQUALS = 12

	def __init__(self, classifier, equations):
		"""
		Construct a new instance of the Grader class with the given classifier
		and input equations.

		Inputs:
		- classifier: TensorFlow Estimator
		- equations: list of numpy arrays forming the input equations

		Return:
		- instance of Grader
		"""
		self.classifier = classifier
		self.equations = equations

	def grade(self, dimen, num_vals, result_file):
		"""
		Grade the equations.

		This is the primary method of the class. It reshapes/formats the input
		images for prediction per number/symbol in the equation. Once each part
		of the equation as been predicted, it then grades the accuracy of the
		overall equation and writes the result to a file.

		Inputs:
		- dimen: tuple of dimensions to reshape the flattened image into
		- num_vals: number of numeric or symbolic values per equation
		- result_file: name of the file to store the results in
		"""
		split_equations = [np.hsplit(np.reshape(
			row, dimen), num_vals) for row in self.equations]
		index = 0
		new_dimen = (dimen[0]**2,)
		equation_arr = self.preprocess_equations(new_dimen, split_equations)
		results = mdl.predict_with_classifier(equation_arr, self.classifier)
		results = [sym["classes"] for sym in results]
		with open(result_file, 'w') as file:
			writer = csv.writer(file)
			for start_index in range(0, len(results), num_vals):
				predicted_values = results[start_index:start_index + num_vals]
				eq_dict = {}
				if predicted_values[1] == self.EQUALS:
					eq_dict["answer"] = predicted_values[0]
					eq_dict["x1"] = predicted_values[2]
					eq_dict["op"] = predicted_values[3]
					eq_dict["x2"] = predicted_values[4]
				elif predicted_values[3] == self.EQUALS:
					eq_dict["answer"] = predicted_values[4]
					eq_dict["x1"] = predicted_values[0]
					eq_dict["op"] = predicted_values[1]
					eq_dict["x2"] = predicted_values[2]
				else:
					print(predicted_values)
					writer.writerow((index, 0))
					index += 1
					continue
				is_correct = self.evaluate_equation(eq_dict)
				writer.writerow((index,int(is_correct)))
				index += 1

	def preprocess_equations(self, dimen, equation_list):
		equation_arr = self.translate_equation(dimen, equation_list[0])
		for equation in equation_list[1:]:
			equation_arr = np.append(equation_arr,
				self.translate_equation(dimen, equation),
				axis=0)
		return equation_arr

	def translate_equation(self, dimen, equation_arr):
		"""
		Translate the separate images of the numeric and symbolic values of an
		equation into flattened arrays, concatenate them into a numpy array, and
		predict each of the values.

		Inputs:
		- dimen: tuple dimension of a flattened piece of the equation
		- equation_arr: list of numpy arrays (one per number/symbol in the equation)

		Return:
		- list of predicted values in the equation
		"""
		for index in range(len(equation_arr)):
			equation_arr[index] = np.reshape(equation_arr[index], dimen)
		equation = np.append([equation_arr[0]], [equation_arr[1]], axis=0)
		for sym in equation_arr[2:]:
			equation = np.append(equation, [sym], axis=0)
		return equation
		# results = mdl.predict_with_classifier(equation, self.classifier)
		# return [sym["classes"] for sym in results]

	def evaluate_equation(self, equation_vals):
		"""
		Evaluate the accuracy of the equation.

		Input:
		- equation_vals: dictionary with the follwing elements:
			- x1: first numeric value
			- x2: second numeric value
			- answer: expected answer to the equation's operation
			- op: equation's operation {+, -}
		
		Return:
		- boolean indicating whether the equation is correct or not
		"""
		x1 = equation_vals["x1"]
		x2 = equation_vals["x2"]
		ans = equation_vals["answer"]
		op = equation_vals["op"]
		if op == self.PLUS:
			return ((x1 + x2) == ans)
		elif op == self.MINUS:
			return ((x1 - x2) == ans)
		else:
			# In case the operation symbol was mis-predicted by the classifier,
			# check if either of the operations return a correct result.
			# Otherwise, return false.
			return ((x1 + x2 == ans) or (x1 - x2 == ans))
