from src.synergy import *
from src.synergy_graph import *
from src.normal_distribution import *

def test_weight_fn_reciprocal():
	assert weight_fn_reciprocal(1) == 1
	assert weight_fn_reciprocal(2) == 0.5

def test_weight_fn_exponential():
	assert weight_fn_exponential(0, 1) == 1
	assert weight_fn_exponential(1, 1) == 2
	assert weight_fn_exponential(5, 1) == 2 ** 5

def test_elementwise_add():
	a_distributions = [NormalDistribution(1,1)]
	b_distributions = [NormalDistribution(4,5)]
	result = elementwise_add(a_distributions, b_distributions)
	assert result == [NormalDistribution(5,6)]

	a_distributions = [NormalDistribution(1,1), NormalDistribution(2,2)]
	b_distributions = [NormalDistribution(4,5), NormalDistribution(6,7)]
	result = elementwise_add(a_distributions, b_distributions)
	assert result == [NormalDistribution(5,6), NormalDistribution(8,9)]

def test_pairwise_synergy():
	# Create a path graph
	G = nx.generators.classic.path_graph(3)
	assert list(G.nodes()) == [0, 1, 2]

	# Create dict of normal distributions for M=2 tasks
	N = dict()
	N[0] = [NormalDistribution(0,1), NormalDistribution(1,2)]
	N[1] = [NormalDistribution(2,3), NormalDistribution(3,4)]
	N[2] = [NormalDistribution(4,5), NormalDistribution(5,6)]

	# Create and query graph
	S = SynergyGraph(G, N)

	# Get pairwise synergy for nodes 0 and 2
	pair_synergy = pairwise_synergy(S, weight_fn_reciprocal, 0, 2)
	assert pair_synergy == [0.5 * NormalDistribution(4,6), 0.5 * NormalDistribution(6,8)]

def test_synergy_with_figure_3():
	# Build graph in figure 3 of paper by Liemhetcharat and Veloso
	G = nx.Graph()
	G.add_edge(1,2)
	G.add_edge(2,4)
	G.add_edge(4,1)
	G.add_edge(4,5)
	G.add_edge(5,6)
	G.add_edge(6,3)

	# Create dict of normal distributions for M=1 tasks
	N = dict()
	N[1] = [NormalDistribution(5,1)]
	N[2] = [NormalDistribution(5,2)]
	N[3] = [NormalDistribution(23,4)]
	N[4] = [NormalDistribution(20,7)]
	N[5] = [NormalDistribution(10,3)]
	N[6] = [NormalDistribution(8,1)]

	# Create graph
	S = SynergyGraph(G, N)

	# Get synergy of different teams described at the end of Section 3.2
	team_A = [1,2]
	team_A_synergy = synergy(S, team_A, weight_fn_reciprocal)
	assert len(team_A_synergy) == 1
	assert team_A_synergy[0].mean == 10

	team_B = [1,2,3]
	team_B_synergy = synergy(S, team_B, weight_fn_reciprocal)
	assert len(team_B_synergy) == 1
	assert team_B_synergy[0].mean == 8

	team_C = [1,2,4]
	team_C_synergy = synergy(S, team_C, weight_fn_reciprocal)
	assert len(team_C_synergy) == 1
	assert team_C_synergy[0].mean == 20




