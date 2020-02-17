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
	synergy = pairwise_synergy(S, weight_fn_reciprocal, 0, 2)
	print('Synergy:')
	for i in synergy:
		print(i)

	print('Expected:')
	for j in [0.5 * NormalDistribution(4,6), 0.5 * NormalDistribution(6,8)]:
		print(j)

	assert synergy == [0.5 * NormalDistribution(4,6), 0.5 * NormalDistribution(6,8)]
