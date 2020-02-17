import numpy as np
import networkx as nx

def pairwise_synergy(S, weight_fn, a, b):
	"""
	Pairwise synergy between two agents a and b
	in a synergy graph S
	"""
	path = nx.algorithms.bidirectional_shortest_path(S.graph , a, b)
	distance = len(path)
	a_distribution = S.get_distributions(a)
	b_distribution = S.get_distributions(b)
	return weight_fn(distance) * (a_distribution + b_distribution)


def weight_fn_reciprocal(d):
	return 1 / d

def weight_fn_exponential(d, h):
	return np.exp(d * np.log(2) / h)