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
	k_max = 100

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
	k_max = 50

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
	
	assert N1 == N2

# def test_estimate_capability_3():
# 	M = 3
# 	mathcal_A = [0,1,2,3,4,5]
# 	k_max = 100

# 	A = [0,1,2]
# 	o1 = [[3,3,3], [4,4,4], [5,5,5]]
# 	observation_group = ObservationGroup(A, M)
# 	observation_group.add_observations(o1)

# 	observation_set = ObservationSet(M, [observation_group])

# 	num_agents = len(mathcal_A)
# 	nearest_neighbors = 3
# 	rewire_prob = 0.3
# 	G = nx.generators.random_graphs.connected_watts_strogatz_graph(num_agents, nearest_neighbors, rewire_prob)
# 	N = estimate_capability(observation_set, G, weight_fn_reciprocal)

# 	keys = [k for k in N.keys()]
# 	assert len(keys) == num_agents
# 	for agent in mathcal_A:
# 		assert len(N[agent]) == M
# 		# print(f"Agent {agent} with performance {N[agent]}")
# 		for distribution in N[agent]:
# 			assert type(distribution) is NormalDistribution
