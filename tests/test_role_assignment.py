from src.role_assignment import *
from src.normal_distribution import *
from src.weight_fns import weight_fn_reciprocal
from tests.test_weighted_synergy_graph import get_figure_2_weighted_synergy_graph

def test_pairwise_synergy_by_role_1():
	WS = get_figure_2_weighted_synergy_graph()
	a = 0
	b = 1
	r_a = 0
	r_b = 1
	assert pairwise_synergy_by_role_policy(WS, weight_fn_reciprocal, a, b, r_a, r_b) == (1/2) * NormalDistribution(3,3)

def test_pairwise_synergy_by_role_2():
	WS = get_figure_2_weighted_synergy_graph()
	a = 0
	b = 0
	r_a = 0
	r_b = 1
	assert pairwise_synergy_by_role_policy(WS, weight_fn_reciprocal, a, b, r_a, r_b) == (1/5) * NormalDistribution(2,3)

def test_pairwise_synergy_by_role_3():
	WS = get_figure_2_weighted_synergy_graph()
	a = 2
	b = 0
	r_a = 0
	r_b = 1
	assert pairwise_synergy_by_role_policy(WS, weight_fn_reciprocal, a, b, r_a, r_b) == (1/6) * NormalDistribution(4,3)

def test_synergy_by_role_policy():
	WS = get_figure_2_weighted_synergy_graph()

	# Assign agent 1 to role 0
	# Assign agent 2 to role 1
	pi = dict()
	pi[0] = 1
	pi[1] = 2

	assert synergy_by_role_policy(WS, pi, weight_fn_reciprocal) == (1/5) * NormalDistribution(5,3)