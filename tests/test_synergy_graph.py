from src.synergy_graph import *
from src.normal_distribution import *

def test_get_distributions():
	# Create path graph that looks like:
	G = nx.generators.classic.path_graph(3)
	assert list(G.nodes()) == [0, 1, 2]

	# Create dict of normal distributions for M=2 tasks
	N = dict()
	N[0] = [NormalDistribution(0,1), NormalDistribution(1,2)]
	N[1] = [NormalDistribution(2,3), NormalDistribution(3,4)]
	N[2] = [NormalDistribution(4,5), NormalDistribution(5,6)]

	# Create and query graph
	S = SynergyGraph(G, N)
	node_0_distributions = S.get_distributions(0)
	assert node_0_distributions == [NormalDistribution(0,1), NormalDistribution(1,2)]

	node_1_distributions = S.get_distributions(1)
	assert node_1_distributions == [NormalDistribution(2,3), NormalDistribution(3,4)]
	assert node_1_distributions != [NormalDistribution(0,1), NormalDistribution(1,2)]

	node_2_distributions = S.get_distributions(2)
	assert node_2_distributions != [NormalDistribution(2,3), NormalDistribution(3,4)]
	assert node_2_distributions != [NormalDistribution(0,1), NormalDistribution(1,2)]
	assert node_2_distributions == [NormalDistribution(4,5), NormalDistribution(5,6)]
