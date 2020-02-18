import itertools
import functools
import numpy as np
import networkx as nx
from scipy.special import comb

def synergy(S, A, weight_fn):
	"""
	S is a SynergyGraph
	A is a list of agents
	"""
	total_pairs = comb(len(A), 2, exact=True)

	pair_synergies = []
	for pair in itertools.combinations(A, r=2):
		pair_synergies.append(pairwise_synergy(S, weight_fn, *pair))

	synergy = functools.reduce(lambda a_distributions, b_distributions : elementwise_add(a_distributions, b_distributions), pair_synergies)
	scale = (1 / total_pairs)
	return list(map(lambda d: scale * d, synergy))

def pairwise_synergy(S, weight_fn, a, b):
	"""
	Pairwise synergy between two agents a and b
	in a synergy graph S
	"""
	path = nx.algorithms.bidirectional_shortest_path(S.graph, a, b)
	distance = len(path) - 1
	a_distribution = S.get_distributions(a)
	b_distribution = S.get_distributions(b)
	sum_distribution = elementwise_add(a_distribution, b_distribution)
	w = weight_fn(distance)
	return list(map(lambda d: w * d, sum_distribution))

def weight_fn_reciprocal(d):
	return 1 / d

def weight_fn_exponential(d, h):
	return np.exp(d * np.log(2) / h)

def elementwise_add(a_distributions, b_distributions):
	assert len(a_distributions) == len(b_distributions)
	result = []
	for i in range(len(a_distributions)):
		result.append(a_distributions[i] + b_distributions[i])
	return result