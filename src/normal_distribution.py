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

	def __rmul__(self, constant):
		assert type(constant) == float or type(constant) == int
		mean = constant * self.mean
		variance = np.power(constant, 2) * self.variance
		return NormalDistribution(mean, variance)

	def __eq__(self, other):
		return (self.mean == other.mean) and (self.variance == other.variance)

	def __str__(self):
		return f"N(m={np.round(self.mean, 2)}, var={np.round(self.variance, 2)})"

	def __repr__(self):
		return str(self)

	def evaluate(self, p):
		"""
		According to Section 3.3, this evaluation balances
		the mean and variance of a normal distribution
		by using a risk factor p in (0,1), where distributions
		with higher variance obtain higher values if p > 0.5
		"""
		return self.mean + np.sqrt(self.variance) * norm.ppf(p)

	def sample(self, size):
		"""
		Sample size points from the normal distribution
		"""
		return np.random.normal(loc=self.mean, scale=np.sqrt(self.variance), size=size)

	def logpdf(self, x):
		"""
		Log likelihood of normal distribution
		"""
		return norm.logpdf(x, loc=self.mean, scale=np.sqrt(self.variance))
