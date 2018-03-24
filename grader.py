"""

@author trevmo
"""

import numpy as np
import model as mdl
import csv


class Grader(object):
	"""
	"""

	PLUS = 10
	MINUS = 11
	EQUALS = 12

	def __init__(self, classifier, equations):
		"""
		"""
		self.classifier = classifier
		self.equations = equations

	def grade(self, dimen, num_vals, result_file):
		split_equations = [np.hsplit(np.reshape(
			row, dimen), num_vals) for row in self.equations]
		index = 0
		new_dimen = (dimen[0]**2,)
		with open(result_file, 'a') as file:
			writer = csv.writer(file)
			for equation in split_equations:
				predicted_values = self.translate_equation(new_dimen, equation)
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
					writer.writerow((index, 0))
					index += 1
					continue
				is_correct = self.evaluate_equation(index, eq_dict)
				writer.writerow((index,int(is_correct)))
				index += 1

	def translate_equation(self, dimen, equation_arr):
		for index in range(len(equation_arr)):
			equation_arr[index] = np.reshape(equation_arr[index], dimen)
		equation = np.append([equation_arr[0]], [equation_arr[1]], axis=0)
		for sym in equation_arr[2:]:
			equation = np.append(equation, [sym], axis=0)
		results = mdl.predict_with_classifier(equation, self.classifier)
		return [sym["classes"] for sym in results]

	def evaluate_equation(self, index, equation_vals):
		x1 = equation_vals["x1"]
		x2 = equation_vals["x2"]
		ans = equation_vals["answer"]
		op = equation_vals["op"]
		if op == self.PLUS:
			return ((x1 + x2) == ans)
		elif op == self.MINUS:
			return ((x1 - x2) == ans)
		else:
			return ((x1 + x2 == ans) or (x1 - x2 == ans))
