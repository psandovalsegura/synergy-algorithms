from src.weighted_synergy_graph import *

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
