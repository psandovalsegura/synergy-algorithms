import itertools
import numpy as np
from scipy.special import comb
from src.normal_distribution import NormalDistribution
from src.synergy_graph import distance_fn

def estimate_capability(O, G, weight_fn):
	"""
	O is an ObservationSet
	G is a networkx graph

	note: there are two implementation differences from the paper:
	1. observation groups are allowed to be of size 1 
	(the paper code would crash with observation groups of size 1 when computing b_variance)
	2. variance cannot be negative after using with least squares solver
	(the paper code allows normal distributions with negative variance after least square solver computation)
	"""
	agents = list(G.nodes)
	num_agents = len(agents)
	num_teams = O.get_num_groups()
	M_mean = np.zeros((num_teams, num_agents))
	M_variance = np.zeros((num_teams, num_agents))
	b_mean = np.zeros((num_teams, 1))
	b_variance = np.zeros((num_teams, 1))

	N = dict()
	for j, a in enumerate(G.nodes):
		N[a] = [None] * O.M

	for m in range(O.M):
		for i, observation_group in enumerate(O.observation_groups):
			for a_j in observation_group.A:
				j = agents.index(a_j)
				M_mean[i][j] = mean_i_j(a_j, observation_group.A, G, weight_fn)
				M_variance[i][j] = variance_i_j(a_j, observation_group.A, G, weight_fn)

			b_mean[i] = (1 / observation_group.size()) * sum(list(map(lambda o: o[m], observation_group.observations)))
			
			observation_group_variance_size = observation_group.size()
			if observation_group_variance_size == 1: observation_group_variance_size = 2
			b_variance[i] = (1 / (observation_group_variance_size - 1)) * sum(list(map(lambda o: (o[m] - b_mean[i]) ** 2, observation_group.observations)))

		means = np.linalg.lstsq(M_mean, b_mean, rcond=None)[0]
		variances = np.linalg.lstsq(M_variance, b_variance, rcond=None)[0]
		for j, a in enumerate(G.nodes):
			N[a][m] = NormalDistribution(means[j].item(), abs(variances[j].item()))

	return N

def mean_i_j(a_j, A_i, G, weight_fn):
	scale = 1 / comb(len(A_i), 2, exact=True)
	summation = 0
	for a in A_i:
		if a != a_j:
			distance = distance_fn(G, a_j, a)
			summation += weight_fn(distance)
	return scale * summation

def variance_i_j(a_j, A_i, G, weight_fn):
	scale = 1 / (comb(len(A_i), 2, exact=True) ** 2)
	summation = 0
	for a in A_i:
		if a != a_j:
			distance = distance_fn(G, a_j, a)
			summation += (weight_fn(distance) ** 2)
	return scale * summation
