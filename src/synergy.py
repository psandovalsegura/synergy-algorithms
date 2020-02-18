import itertools
import functools
import random
import numpy as np
import networkx as nx
from scipy.special import comb

def create_synergy_graph(O, mathcal_A):
	"""
	O is an observation set
	mathcal_A is the set of all agents
	"""
	num_agents = len(mathcal_A)
	nearest_neighbors = 3
	rewire_prob = 0.30
	G = nx.generators.random_graphs.connected_watts_strogatz_graph(num_agents, nearest_neighbors, rewire_prob)

	pass

def get_approx_optimal_team_brute(S, mathcal_A, p):
	"""
	S is a SynergyGraph
	mathcal_A is the set of all agents
	p is the risk factor
	"""
	num_agents = len(mathcal_A)
	best_value = -1
	best_team = None
	for n in range(num_agents):
		team, value = get_approx_optimal_team(S, n, p)
		if value > best_value:
			best_value = value
			best_team = team
	return best_team

def get_approx_optimal_team(S, mathcal_A, n, p, weight_fn):
	"""
	S is a SynergyGraph
	mathcal_A is the set of all agents
	n is the optimal size of the team, if unknown use brute force version
	p is the risk factor
	"""
	A = random.sample(mathcal_A, n)
	distributions = synergy(S, A, weight_fn)
	value = value_fn_sum(distributions, p)
	pass

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

def value_fn_sum(distributions, p):
	return sum(list(map(lambda distr: distr.evaluate(p), distributions)))

def elementwise_add(a_distributions, b_distributions):
	assert len(a_distributions) == len(b_distributions)
	result = []
	for i in range(len(a_distributions)):
		result.append(a_distributions[i] + b_distributions[i])
	return result