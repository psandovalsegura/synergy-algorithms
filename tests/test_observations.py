import networkx as nx
from src.observation_set import *
from src.observations import *
from src.synergy_graph import display
from src.normal_distribution import NormalDistribution
from src.synergy import weight_fn_reciprocal

def test_estimate_capability_1():
	"""
	check that every agent has M normal distributions
	"""
	M = 1
	mathcal_A = [0,1,2,3]

	A = [0,1,2]
	o1 = [[3], [4], [5]]
	observation_group = ObservationGroup(A, M)
	observation_group.add_observations(o1)

	observation_set = ObservationSet(M, [observation_group])

	num_agents = len(mathcal_A)
	nearest_neighbors = 3
	rewire_prob = 0.3
	G = nx.generators.random_graphs.connected_watts_strogatz_graph(num_agents, nearest_neighbors, rewire_prob)
	N = estimate_capability(observation_set, G, weight_fn_reciprocal)

	keys = [k for k in N.keys()]
	assert len(keys) == num_agents
	for agent in mathcal_A:
		assert len(N[agent]) == M
		for distribution in N[agent]:
			assert type(distribution) is NormalDistribution

def test_estimate_capability_2():
	"""
	check that the capabilities of agents don't 
	change if the synergy graph doesn't change
	"""
	M = 1
	mathcal_A = [0,1,2,3]

	A = [0,1,2]
	o1 = [[3], [4], [5]]
	observation_group = ObservationGroup(A, M)
	observation_group.add_observations(o1)

	A2 = [0,3]
	o2 = [[30], [40], [30], [35]]
	observation_group2 = ObservationGroup(A2, M)
	observation_group2.add_observations(o2)

	observation_set = ObservationSet(M, [observation_group, observation_group2])

	num_agents = len(mathcal_A)
	nearest_neighbors = 3
	rewire_prob = 0.30
	G = nx.generators.random_graphs.connected_watts_strogatz_graph(num_agents, nearest_neighbors, rewire_prob)
	initial_edges = [e for e in G.edges]

	N1 = estimate_capability(observation_set, G, weight_fn_reciprocal)

	G_prime = G = nx.Graph()
	for edge in initial_edges:
		G_prime.add_edge(*edge)

	N2 = estimate_capability(observation_set, G_prime, weight_fn_reciprocal)
	
	for i in mathcal_A:
		assert np.round(N1[i][0].mean, 3) == np.round(N2[i][0].mean, 3)
		assert np.round(N1[i][0].variance, 3) == np.round(N2[i][0].variance, 3)

def test_estimate_capability_3():
	"""
	consider all possible teams each with the same performance
	and check that agent capabilities are equal
	"""
	M = 1
	mathcal_A = [0,1,2]

	A1 = [0,1]
	o1 = [[30], [40], [40], [30]]
	observation_group = ObservationGroup(A1, M)
	observation_group.add_observations(o1)

	A2 = [0,2]
	o2 = [[30], [40], [40], [30]]#[[3], [4], [3]]
	observation_group2 = ObservationGroup(A2, M)
	observation_group2.add_observations(o2)

	A3 = [0,1,2]
	o3 = [[30], [40], [40], [30]]#[[10], [15], [10]]
	observation_group3 = ObservationGroup(A3, M)
	observation_group3.add_observations(o3)

	A4 = [1,2]
	o4 = [[30], [40], [40], [30]]#[[7], [8], [9]]
	observation_group4 = ObservationGroup(A4, M)
	observation_group4.add_observations(o4)

	observation_set = ObservationSet(M, [observation_group, observation_group2, observation_group3, observation_group4])

	num_agents = len(mathcal_A)
	nearest_neighbors = 3
	rewire_prob = 0.3
	G = nx.generators.random_graphs.connected_watts_strogatz_graph(num_agents, nearest_neighbors, rewire_prob)
	N = estimate_capability(observation_set, G, weight_fn_reciprocal)

	keys = [k for k in N.keys()]
	common_capability_mean = np.round(N[0][0].mean, 3)
	common_capability_variance = np.round(N[0][0].variance, 3)
	assert common_capability_mean > 0
	assert common_capability_variance > 0
	assert len(keys) == num_agents
	for agent in mathcal_A:
		assert len(N[agent]) == M
		assert np.round(N[agent][0].mean, 3) == common_capability_mean
		assert np.round(N[agent][0].variance, 3) == common_capability_variance

def test_estimate_capability_4():
	"""
	check that we can estimate capability with an observation 
	group of size 1 (no asserts, just checking that there are no errors)
	"""
	M = 1
	mathcal_A = [0,1,2,3]

	A = [0,1,2]
	o1 = [[3]]
	observation_group = ObservationGroup(A, M)
	observation_group.add_observations(o1)

	A2 = [0,3]
	o2 = [[30], [40], [30], [35]]
	observation_group2 = ObservationGroup(A2, M)
	observation_group2.add_observations(o2)

	observation_set = ObservationSet(M, [observation_group, observation_group2])

	num_agents = len(mathcal_A)
	nearest_neighbors = 3
	rewire_prob = 0.30
	G = nx.generators.random_graphs.connected_watts_strogatz_graph(num_agents, nearest_neighbors, rewire_prob)
	initial_edges = [e for e in G.edges]

	N1 = estimate_capability(observation_set, G, weight_fn_reciprocal)

