import itertools
import functools
import random
import numpy as np
import networkx as nx
from src.annealing import annealing
from scipy.special import comb
from scipy.optimize import basinhopping

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

def get_approx_optimal_team_brute(S, mathcal_A, p, k_max, weight_fn):
	"""
	S is a SynergyGraph
	mathcal_A is the set of all agents
	p is the risk factor
	"""
	num_agents = len(mathcal_A)
	best_value = -1
	best_team = None
	for n in range(1, num_agents + 1):
		team, value = get_approx_optimal_team(S, mathcal_A, n, p, k_max, weight_fn)
		if value > best_value:
			best_value = value
			best_team = team
	return best_team

class RandomTeamNeighborStep(object):
	def __init__(self, S, mathcal_A, A, weight_fn):
		"""
		S is a SynergyGraph
		mathcal_A is the set of all agents
		A is a list of agents
		"""
		self.S = S
		self.mathcal_A = mathcal_A
		self.A = A
		self.weight_fn = weight_fn

	def __call__(self, x):
		"""
		x is list of distributions
		"""
		new_A = random_team_neighbor(self.mathcal_A, self.A)
		distributions = synergy(self.S, new_A, self.weight_fn)
		self.A = new_A
		return distributions

def get_approx_optimal_team(S, mathcal_A, n, p, weight_fn, k_max):
	"""
	S is a SynergyGraph
	mathcal_A is the set of all agents
	n is the optimal size of the team, if unknown use brute force version
	p is the risk factor
	"""
	initial_team = random.sample(mathcal_A, n)
	value_function = lambda x: value_fn_sum(synergy(S, x, weight_fn), p)
	random_neighbor = lambda a: random_team_neighbor(mathcal_A, a)

	final_team, final_value, teams, values = annealing(initial_team, value_function, random_neighbor, debug=True, maxsteps=k_max)
	return final_team, final_value, teams, values

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

def random_team_neighbor(mathcal_A, A):
	"""
	mathcal_A is the set of all agents
	A is a list of agents

	note: this function does not support 
	multiple agents of the same type
	"""
	if len(mathcal_A) == 0:
		raise RuntimeError('There are no agents available, mathcal_A is empty')

	mathcal_A_set = set(mathcal_A)
	A_set = set(A)
	difference = mathcal_A_set - A_set

	# if A contains all the available agents
	# then nothing can be done
	if len(difference) == 0:
		return A

	new_agent = random.choice(list(difference))
	random_index = random.choice(range(len(A)))
	B = A.copy()
	B[random_index] = new_agent
	return B

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