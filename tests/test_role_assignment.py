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

def test_create_random_role_assignment_1():
	"""
	check that the role assignment fills all the roles
	Note: roles must be indexed starting at 0
	"""
	pi = create_random_role_assignment(6, [0,1,2,3])
	assert len(pi.keys()) == 4
	for key in pi.keys():
		assert pi[key] <= 6


def test_create_random_role_assignment_2():
	"""
	check that the role assignment fills all the roles
	Note: roles must be indexed starting at 0
	"""
	pi = create_random_role_assignment(10, [0,1,2,3])
	assert len(pi.keys()) == 4
	for key in pi.keys():
		assert pi[key] <= 10

def test_create_random_role_assignment_3():
	"""
	check that the role assignment fills all the roles
	Note: roles must be indexed starting at 0
	"""
	pi = create_random_role_assignment(2, [0,1,2,3,4,5])
	assert len(pi.keys()) == 6
	for key in pi.keys():
		assert pi[key] <= 2

def test_random_role_assignment_neighbor():
	"""
	check that only one change is made to the role assignment policy
	"""
	num_agents = 10
	R = [0,1,2,3]
	pi = create_random_role_assignment(num_agents, R)
	pi_prime = random_role_assignment_neighbor(pi, num_agents, R)

	assert len(pi.keys()) == len(pi_prime.keys())
	num_changes = 0
	for key in pi_prime.keys():
		if pi_prime[key] != pi[key]:
			num_changes += 1

	assert num_changes == 1

def test_log_likelihood_by_role():
	"""
	check that the log likelihood of the training data is correct 
	"""
	WS = get_figure_2_weighted_synergy_graph()

	pi = dict()
	pi[0] = 1
	pi[1] = 2
	v = 1.5
	T = [(pi, v)]

	expected_synergy = (1/5) * NormalDistribution(5,3)
	expected_log_likelihood = -0.5 * np.log(2 * np.pi * expected_synergy.variance) - 0.5 * ((v - expected_synergy.mean)**2/(expected_synergy.variance))

	assert np.round(log_likelihood_by_role(WS, T, weight_fn_reciprocal), 3) == np.round(expected_log_likelihood, 3)




