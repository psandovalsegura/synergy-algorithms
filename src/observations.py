import itertools
import numpy as np
from scipy.special import comb
from src.normal_distribution import NormalDistribution
from src.synergy_graph import distance_fn

def estimate_capability(O, G, weight_fn):
	"""
	O is an ObservationSet
	G is a networkx graph
	"""
	num_agents = len(G.nodes)
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
			for j, a_j in enumerate(observation_group.A):
				M_mean[i][j] = mean_i_j(a_j, observation_group.A, G, weight_fn)
				M_variance[i][j] = variance_i_j(a_j, observation_group.A, G, weight_fn)

			b_mean[i] = (1 / len(observation_group.A)) * sum(list(map(lambda o: o[m], observation_group.observations)))
			b_variance[i] = (1 / (len(observation_group.A) - 1)) * sum(list(map(lambda o: (o[m] - b_mean[i]) ** 2, observation_group.observations)))

		means = np.linalg.lstsq(M_mean, b_mean, rcond=None)[0]
		variances = np.linalg.lstsq(M_variance, b_variance, rcond=None)[0]
		for j, a in enumerate(G.nodes):
			N[a][m] = NormalDistribution(means[j], variances[j])

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
