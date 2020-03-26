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
	check that the log likelihood of the training data follows the formula in Section 5.B.
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

def test_get_means_coefficient_row_1():
	"""
	check simple example where synergy consists of only a single 
	pairwise synergy computation
	"""
	WS = get_figure_2_weighted_synergy_graph()

	R = [0, 1]
	pi = dict()
	pi[0] = 0
	pi[1] = 1

	row = get_means_coefficient_row(WS.graph, pi, weight_fn_reciprocal)

	assert len(row) == len(R) * len(WS.graph.nodes)
	assert row[0] == 0.5 and row[3] == 0.5
	assert row[1] == 0 and row[2] == 0 and row[4] == 0 and row[5] == 0

def test_get_means_coefficient_row_2():
	"""
	a more complicated coefficient row with an additional role
	"""
	WS = get_figure_2_weighted_synergy_graph()

	R = [0, 1, 2]
	pi = dict()
	pi[0] = 0
	pi[1] = 1
	pi[2] = 1

	row = get_means_coefficient_row(WS.graph, pi, weight_fn_reciprocal)
	assert len(row) == len(R) * len(WS.graph.nodes)
	assert row[0] == (1/3) * (1/2 + 1/2)
	assert row[4] == (1/3) * (1/2 + 1/4)
	assert row[5] == (1/3) * (1/2 + 1/4)

def test_estimate_means_by_role():
	"""
	check that the means for a single training example make sense

	Note: for a 3 agent team with 2 roles the means vector is arranged 
	in the following way
	[[mu_agent=0_role=0]
	[mu_agent=0_role=1]
	[mu_agent=1_role=0]
	[mu_agent=1_role=1]
	[mu_agent=2_role=0]
	[mu_agent=2_role=1]]
	"""
	WS = get_figure_2_weighted_synergy_graph()

	R = [0, 1]
	pi = dict()
	pi[0] = 0
	pi[1] = 1
	v = 5
	T = [(pi, v)]

	means = estimate_means_by_role(WS.graph, R, T, weight_fn_reciprocal)
	assert np.round(means[0], 3) == 5
	assert np.round(means[1 * len(R) + 1], 3) == 5

def test_estimate_capability_by_role_1():
	"""
	check that the capability dictionary is properly instantiated

	Note: variance of distributions is disregarded here
	"""
	WS = get_figure_2_weighted_synergy_graph()

	R = [0, 1]
	pi = dict()
	pi[0] = 0
	pi[1] = 1
	v = 5
	T = [(pi, v)]

	C = estimate_capability_by_role(WS.graph, R, T, weight_fn_reciprocal)
	assert len(C.keys()) == len(WS.graph.nodes)
	assert np.round(C[0][0].mean, 3) == 5 and np.round(C[0][1].mean, 3) == 0
	assert np.round(C[1][0].mean, 3) == 0 and np.round(C[1][1].mean, 3) == 5

def test_learn_weighted_synergy_graph_1():
	"""
	sanity check that the function doesn't crash
	"""
	R = [0, 1]
	pi = dict()
	pi[0] = 0
	pi[1] = 1
	v = 5
	T = [(pi, v)]
	num_agents = 3
	k_max = 200

	final_sgraph, final_value, sgraphs, values = learn_weighted_synergy_graph(num_agents, R, T, weight_fn_reciprocal, k_max, display=True)
	assert len(sgraphs) == len(values)
	assert len(values) <= k_max
	assert final_value == values[-1]

def test_learn_weighted_synergy_graph_2():
	"""
	check that we can learn the graph in Figure 2 with synthetic training examples
	"""
	pass


def test_learn_weighted_synergy_graph_3():
	"""
	check that we can learn an arbitrary graph with synthetic training examples
	"""
	pass

