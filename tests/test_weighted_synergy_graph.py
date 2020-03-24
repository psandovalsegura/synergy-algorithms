from src.weighted_synergy_graph import *
from src.normal_distribution import *

def get_figure_2_weighted_synergy_graph():
	"""
	build the graph in figure 2 of WeSGRA paper by Liemhetcharat and Veloso
	"""
	G = nx.Graph()
	G.add_edge(0, 0, weight=5)
	G.add_edge(1, 1, weight=4)
	G.add_edge(2, 2, weight=1)
	G.add_edge(0, 1, weight=2)
	G.add_edge(1, 2, weight=5)
	G.add_edge(2, 0, weight=6)

	# Create dict of normal distributions for M=2 roles
	# Note: distributions here are cosmetic, since the paper doesn't specify
	N = dict()
	N[0] = [NormalDistribution(1,1), NormalDistribution(1,2)]
	N[1] = [NormalDistribution(2,1), NormalDistribution(2,2)]
	N[2] = [NormalDistribution(3,1), NormalDistribution(3,2)]

	# Create graph
	WS = WeightedSynergyGraph(G, N)
	return WS

def test_get_distance_1():
	"""
	check that distance to and from the same node 
	have the weight of the self-loop
	"""
	G = nx.Graph()
	G.add_edge(0, 0, weight=4.1)
	G.add_edge(0, 1, weight=4.2)
	G.add_edge(1, 1, weight=4.3)

	WS = WeightedSynergyGraph(G, dict())

	assert WS.get_distance(0, 0) == 4.1
	assert WS.get_distance(1, 1) == 4.3

def test_get_distance_2():
	"""
	check that distance to and from the same node 
	have the weight of the self-loop
	"""
	G = nx.Graph()
	G.add_edge(0, 0, weight=4.1)
	G.add_edge(0, 1, weight=4.2)
	G.add_edge(1, 1, weight=4.3)

	WS = WeightedSynergyGraph(G, dict())

	assert WS.get_distance(0, 1) == 4.2

def test_get_distance_3():
	"""
	check that distance to and from a node in Figure 2 is correct
	"""
	WS = get_figure_2_weighted_synergy_graph()

	assert WS.get_distance(0, 0) == 5
	assert WS.get_distance(1, 1) == 4
	assert WS.get_distance(2, 2) == 1
	assert WS.get_distance(2, 1) == 5
	assert WS.get_distance(2, 0) == 6
	assert WS.get_distance(0, 1) == 2

def test_get_capability():
	"""
	check that the correct distribution is returned
	"""
	WS = get_figure_2_weighted_synergy_graph()
	assert WS.get_capability(0, 0) == NormalDistribution(1,1)
	assert WS.get_capability(0, 1) == NormalDistribution(1,2)
	assert WS.get_capability(1, 0) == NormalDistribution(2,1)
	assert WS.get_capability(1, 1) == NormalDistribution(2,2)
	assert WS.get_capability(2, 0) == NormalDistribution(3,1)
	assert WS.get_capability(2, 1) == NormalDistribution(3,2)

def test_random_weighted_graph_neighbor_1():
	"""
	check that the weights on the graph are changed properly
	"""
	random.seed(1)
	WS = get_figure_2_weighted_synergy_graph()
	G = WS.graph

	# Based on this seed, the first change increases the edge weight
	H = random_weighted_graph_neighbor(G)
	assert H[0][0]['weight'] == 6

	# Based on this seed, the second change decreases the edge weight
	H = random_weighted_graph_neighbor(G)
	assert H[1][1]['weight'] == 3

	initial_nodes = [n for n in G]
	initial_edges = [e for e in G.edges]

	new_nodes = [n for n in H]
	new_edges = [e for e in H.edges]

	assert new_nodes == initial_nodes
	assert new_edges == initial_edges
	

