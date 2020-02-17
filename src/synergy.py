import numpy as np

def pairwise_synergy(S, a, b):
	"""
	Pairwise synergy between two agents a and b
	in a synergy graph S
	"""
	pass


def weight_fn_reciprocal(d):
	return 1 / d

def weight_fn_exponential(d, h):
	return np.exp(d * np.log(2) / h)