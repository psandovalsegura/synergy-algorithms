import pytest
import itertools
from src.synergy import *
from src.synergy_graph import *
from src.normal_distribution import *
from src.observation_set import *

def get_figure_3_synergy_graph():
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
	return S

def test_weight_fn_reciprocal():
	assert weight_fn_reciprocal(1) == 1
	assert weight_fn_reciprocal(2) == 0.5

def test_weight_fn_exponential():
	assert weight_fn_exponential(0, 1) == 1
	assert weight_fn_exponential(1, 1) == 2
	assert weight_fn_exponential(5, 1) == 2 ** 5

def test_value_fn_sum():
	p = 0.5
	a_distributions = [NormalDistribution(1,1), NormalDistribution(2,2)]
	a_0_eval = NormalDistribution(1,1).evaluate(p)
	a_1_eval = NormalDistribution(2,2).evaluate(p)
	assert value_fn_sum(a_distributions, p) == (a_0_eval + a_1_eval)

	b_distributions = [NormalDistribution(4,5), NormalDistribution(6,7)]
	b_0_eval = NormalDistribution(4,5).evaluate(p)
	b_1_eval = NormalDistribution(6,7).evaluate(p)
	assert value_fn_sum(b_distributions, p) == (b_0_eval + b_1_eval)

def test_value_fn_with_synergy_repeat():
	S = get_figure_3_synergy_graph()
	p = 0.5
	f = lambda team: value_fn_sum(synergy(S, team, weight_fn_reciprocal), p)

	A1 = [5,2,6]
	value_A1 = f(A1)
	A2 = [2,5,6]
	value_A2 = f(A2)
	A3 = [6,2,5]
	value_A3 = f(A3)

	assert value_A1 == value_A2
	assert value_A2 == value_A3
	assert value_A1 == value_A3

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
	S = get_figure_3_synergy_graph()

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

	# Check final task values
	p = 0.50
	team_A_value = team_A_synergy[0].evaluate(p)
	team_B_value = team_B_synergy[0].evaluate(p)
	team_C_value = team_C_synergy[0].evaluate(p)
	print(team_A_value, team_B_value, team_C_value)
	assert team_A_value > team_B_value
	assert team_C_value > team_B_value
	assert team_C_value > team_A_value

def test_random_team_neighbor_full_team():
	mathcal_A = [1,2,3,4]
	A = [1,2,3,4]
	B = random_team_neighbor(mathcal_A, A)
	assert A == B

def test_random_team_neighbor_1():
	mathcal_A = [1,2,3,4]
	A = [1,2,3]
	B = random_team_neighbor(mathcal_A, A)
	assert len(B) == 3
	assert (4 in B)

def test_random_team_neighbor_2():
	mathcal_A = [4,3,2,1]
	A = [1,2]
	B = random_team_neighbor(mathcal_A, A)
	assert len(B) == 2
	assert (1 not in B) or (2 not in B)

def test_random_team_neighbor_3():
	mathcal_A = [4,3,2,1]
	A = [1,2]
	original_A = A.copy()
	B = random_team_neighbor(mathcal_A, A)
	assert len(A) == 2
	assert A == original_A

def test_random_team_neighbor_runtime_err():
	mathcal_A = []
	A = [1,2,3,4]
	with pytest.raises(RuntimeError) as excinfo:
		B = random_team_neighbor(mathcal_A, A)
	assert "no agents available" in str(excinfo.value)

def test_get_approx_optimal_team_figure_3():
	S = get_figure_3_synergy_graph()
	mathcal_A = [1,2,3,4,5,6]
	n = 3
	p = 0.50
	k_max = 100
	approx_A, approx_value, approx_teams, approx_values = get_approx_optimal_team(S, mathcal_A, n, p, k_max, weight_fn_reciprocal)

	# Optimal teams of size 3 (to test the solution found by annealing)
	# [3,5,6] with value 21.8333
	# [1,4,5] with value 20.8333
	# [2,4,5] with value 20.8333
	# [4,5,6] with value 20.6666
	# [3,4,5] with value 20.2777

	found_team_1 = set(approx_A) == set([3,5,6]) and np.round(approx_value, 3) == 21.833
	found_team_2 = set(approx_A) == set([1,4,5]) and np.round(approx_value, 3) == 20.833
	found_team_3 = set(approx_A) == set([2,4,5]) and np.round(approx_value, 3) == 20.833
	
	assert found_team_1 or found_team_2 or found_team_3
	assert len(approx_teams) == len(approx_values)
	assert len(approx_values) <= k_max

def test_get_approx_optimal_team_brute_figure_3():
	S = get_figure_3_synergy_graph()
	mathcal_A = [1,2,3,4,5,6]
	p = 0.50
	k_max = 100
	brute_best_team, brute_best_value = get_approx_optimal_team_brute(S, mathcal_A, p, k_max, weight_fn_reciprocal)

	# Optimal teams (to test the solution found by annealing)
	# [3,6] with value 31.0
	# [4,5] with value 30.0
	# [1,4] with value 25.0
	# [2,4] with value 25.0
	# [3,5,6] with value 21.8333

	found_team_1 = set(brute_best_team) == set([3,6]) and np.round(brute_best_value, 3) == 31.0
	found_team_2 = set(brute_best_team) == set([4,5]) and np.round(brute_best_value, 3) == 30.0
	found_team_3 = set(brute_best_team) == set([1,4]) and np.round(brute_best_value, 3) == 25.0

	assert found_team_1 or found_team_2 or found_team_3

def test_random_graph_neighbor_1():
	G = nx.generators.classic.path_graph(6)
	initial_nodes = [n for n in G]
	initial_edges = [e for e in G.edges]

	random_graph_neighbor(G)
	new_nodes = [n for n in G]
	new_edges = [e for e in G.edges]

	assert len(new_nodes) == len(initial_nodes)
	assert (len(new_edges) - 1) == len(initial_edges) or (len(new_edges) + 1) == len(initial_edges)
	assert new_edges != initial_edges
	assert new_nodes == initial_nodes

def test_random_graph_neighbor_2():
	G = nx.Graph()
	G.add_edge(1,2)
	G.add_edge(2,4)
	G.add_edge(4,1)
	G.add_edge(4,5)
	G.add_edge(5,6)
	G.add_edge(6,3)

	initial_nodes = [n for n in G]
	initial_edges = [e for e in G.edges]

	H = random_graph_neighbor(G)
	new_nodes = [n for n in H]
	new_edges = [e for e in H.edges]

	assert len(new_nodes) == len(initial_nodes)
	assert (len(new_edges) - 1) == len(initial_edges) or (len(new_edges) + 1) == len(initial_edges)
	assert new_edges != initial_edges
	assert new_nodes == initial_nodes

def test_random_graph_neighbor_3():
	"""
	check that the synergy graph can be iterated with 
	random graph neighbor
	"""
	S = get_figure_3_synergy_graph()
	initial_nodes = [n for n in S.graph.nodes]
	initial_edges = [e for e in S.graph.edges]

	random_graph_neighbor(S.graph)

	new_nodes = [n for n in S.graph.nodes]
	new_edges = [e for e in S.graph.edges]

	assert len(new_nodes) == len(initial_nodes)
	assert (len(new_edges) - 1) == len(initial_edges) or (len(new_edges) + 1) == len(initial_edges)
	assert new_edges != initial_edges
	assert new_nodes == initial_nodes

def test_log_likelihood_w_new_graph():
	"""
	check that the log likelihood of observations changes
	with a random graph neighbor of a synergy graph
	"""
	S = get_figure_3_synergy_graph()
	M = 1
	A = [1,2,3]
	o1 = [[3], [4], [5]]
	observation_group = ObservationGroup(A, M)
	observation_group.add_observations(o1)

	A2 = [3,4]
	o2 = [[30], [40], [30], [35]]
	observation_group2 = ObservationGroup(A2, M)
	observation_group2.add_observations(o2)
	observation_set = ObservationSet(M, [observation_group, observation_group2])
	
	loglike = log_likelihood(observation_set, S, weight_fn_reciprocal)
	random_graph_neighbor(S.graph)
	loglike2 = log_likelihood(observation_set, S, weight_fn_reciprocal)
	assert loglike != loglike2

# def test_create_synergy_graph():
# 	M = 1
# 	mathcal_A = [0,1,2,3]
# 	k_max = 50

# 	A = [0,1,2]
# 	o1 = [[3], [4], [5]]
# 	observation_group = ObservationGroup(A, M)
# 	observation_group.add_observations(o1)

# 	A2 = [0,3]
# 	o2 = [[30], [40], [30], [35]]
# 	observation_group2 = ObservationGroup(A2, M)
# 	observation_group2.add_observations(o2)

# 	observation_set = ObservationSet(M, [observation_group, observation_group2])

# 	final_sgraph, final_value, sgraphs, values = create_synergy_graph(observation_set, mathcal_A, weight_fn_reciprocal, k_max, display=True)
# 	print(f"SynergyGraphs: {sgraphs}")
# 	print(f"Values: {values}")
# 	print(f"Final SynergyGraph: {final_sgraph} Value: {final_value}")
# 	# todo: figure out why variances are negative
# 	# assert 0
