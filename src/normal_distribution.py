from scipy.stats import norm
import numpy as np

class NormalDistribution:
	def __init__(self, mean, variance):
		self.mean = mean
		self.variance = variance

	def __add__(self, other):
		mean = self.mean + other.mean
		variance = self.variance + other.variance
		return NormalDistribution(mean, variance)

	def __eq__(self, other):
		return (self.mean == other.mean) and (self.variance == other.variance)

	def __str__(self):
		return "NormalDistribution({0}, {1})".format(self.mean, self.variance)

	def evaluate(self, p):
		return self.mean + np.sqrt(self.variance) * norm.ppf(p)
