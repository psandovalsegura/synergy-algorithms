import pickle
import pytest
from src.role_assignment import *
from src.normal_distribution import *
from src.observation_set import *
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

# Test case turned off (due to runtime), but it has passed!
def off_test_learn_weighted_synergy_graph_1():
	"""
	sanity check that the simulated annealing returns appropriate list of graphs and values
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

def create_role_policy_for_roomba_drone_team(A, team_size):
	"""
	Agents 0 - 4 are roomba
	Agents 5 - 10 are drone
	"""
	pi = dict()
	for i in range(team_size):
		# For role i of 5, what agent was chosen?
		# Agent index 0 is roomba, agent index 1 is drone
		agent = A[i]
		agent_index = 0 if agent in range(team_size) else 1
		pi[i] = agent_index
	return pi

def get_roomba_ratio(team_size, final_pi):
	num_drones = sum(final_pi.values())
	num_roomba = team_size - num_drones
	return num_roomba / team_size

# Test turned off, but could be run by deleting the 'off_' prefix then
# executing tests marked 'slow' with 'python -m pytest -vv tests/ -m slow'
@pytest.mark.slow
def off_test_learn_weighted_synergy_graph_and_get_role_assignment_1():
	sensor_radius_list = [15]
	max_time_list = [2008]
	set_size_list_of_lists = [[107]]
	true_size_list = [100]

	team_size = 10
	environment_width = 50
	runs_to_average = 1

	for sensor_radius_index, sensor_radius in enumerate(sensor_radius_list):
		max_time = max_time_list[sensor_radius_index]
		set_size_list = set_size_list_of_lists[sensor_radius_index]
		result_str = ""
		for set_size_index, set_size in enumerate(set_size_list):
			ts = true_size_list[set_size_index]

			directory = "/Users/pedrosandoval/Documents/UMDReasearch/synergy-algorithms-data/roomba-drone-teams/"
			filename = f"{directory}new_observation_set_team_size={team_size}_env={environment_width}_sensorradius={sensor_radius}_maxtime={max_time}_setsize={set_size}"
			observation_set = pickle.load(open(filename, "rb"))


			result_str += f"Num observation groups (set_size): {observation_set.get_num_groups()}\n"

			# There was only M=1 subtask when this observation set was collected
			# but there are 24 observation groups with 10 roomba and drone
			# Note: since we need |T| > 2 N*M = 2 * 2 * 10
			T = [] 
			for m in range(observation_set.M):
				for i, observation_group in enumerate(observation_set.observation_groups):
					# Only create a training example if within the true size
					if i >= ts:
						break
					pi = create_role_policy_for_roomba_drone_team(observation_group.A, team_size)
					v = (1 / len(observation_group.observations)) * sum(list(map(lambda o: o[m], observation_group.observations)))
					T.append((pi, v))

			result_str += f"Num training examples (true size): {len(T)}\n"
			roomba_ratios = []

			for rta in range(runs_to_average):
				# 2 agents: roomba and drone
				# 5 roles: since the scaled team sizes are of size 5
				R = list(range(team_size))
				num_agents = 2
				M = 1
				k_max = 200


				w_min = 1
				w_max = 10
				G = nx.generators.classic.path_graph(2)
				for edge in G.edges:
					G.edges[edge]['weight'] = random.choice(range(w_min, w_max + 1))

				G.add_edge(0, 0, weight=random.choice(range(w_min, w_max + 1)))
				G.add_edge(1, 1, weight=random.choice(range(w_min, w_max + 1)))

				final_sgraph, final_value, sgraphs, values = learn_weighted_synergy_graph(num_agents, R, T, weight_fn_reciprocal, k_max, G=G, display=False)	

				# Get role assignment policy
				p = 0.5
				final_pi, final_value, pis, values = get_approx_optimal_role_assignment_policy(final_sgraph, R, p, weight_fn_reciprocal, k_max)
				roomba_ratios.append(get_roomba_ratio(team_size, final_pi))

			result_str += f"Final graph:\n{final_sgraph}\n"
			result_str += f"pis:{pis}\n"
			result_str += f"values:{values}\n"
			result_str += f"final_pi:{final_pi}\n"
			result_str += f"final value:{final_value}\n"
			result_str += f"avg roomba ratio (out of {runs_to_average} runs):{sum(roomba_ratios) / runs_to_average}\n"
			result_str += f"From data in: {filename}\n\n\n"


		with open(f"single_run_results_team_size={team_size}_env={environment_width}_sensorradius={sensor_radius}.txt", "w") as f:
			f.write(result_str)

# Test turned off, but could be run by deleting the 'off_' prefix then
# executing tests marked 'slow' with 'python -m pytest -vv tests/ -m slow'
@pytest.mark.slow
def off_test_learn_weighted_synergy_graph_and_get_role_assignment_AVG():
	"""
	train a weighted synergy graph on roomba-drone team data 
	and print what the model thinks is the best role assignment
	into files

	average all results over a number of runs

	Note: the team size will be considered the number of roles available
	"""
	
	sensor_radius_list = [15,14,13,12,11,10,9]
	max_time_list = [2008,1922,1874,1778,1845,2309,2005]
	set_size_list_of_lists = [[30,59,117,175,233,291,1159], [30,59,117,175,233,291,1159], [30,59,117,175,233,291,1157], [30,59,117,175,233,291,1159], [30,59,117,175,233,291,1158], [30,59,117,175,233,291,1160], [30,59,117,175,233,291,1161]]
	true_size_list = [30,50,100,150,200,250,300]

	team_size = 30
	environment_width = 50
	runs_to_average = 10

	for sensor_radius_index, sensor_radius in enumerate(sensor_radius_list):
		max_time = max_time_list[sensor_radius_index]
		set_size_list = set_size_list_of_lists[sensor_radius_index]
		result_str = ""
		for set_size_index, set_size in enumerate(set_size_list):
			ts = true_size_list[set_size_index]

			directory = "/Users/pedrosandoval/Documents/UMDReasearch/synergy-algorithms-data/roomba-drone-teams/"
			filename = f"{directory}observation_set_team_size={team_size}_env={environment_width}_sensorradius={sensor_radius}_maxtime={max_time}_setsize={set_size}"
			observation_set = pickle.load(open(filename, "rb"))


			result_str += f"Num observation groups (set_size): {observation_set.get_num_groups()}\n"

			# There was only M=1 subtask when this observation set was collected
			# but there are 24 observation groups with 10 roomba and drone
			# Note: since we need |T| > 2 N*M = 2 * 2 * 10
			T = [] 
			for m in range(observation_set.M):
				for i, observation_group in enumerate(observation_set.observation_groups):
					# Only create a training example if within the true size
					if i >= ts:
						break
					pi = create_role_policy_for_roomba_drone_team(observation_group.A, team_size)
					v = (1 / len(observation_group.observations)) * sum(list(map(lambda o: o[m], observation_group.observations)))
					T.append((pi, v))

			result_str += f"Num training examples (true size): {len(T)}\n"
			roomba_ratios = []

			for rta in range(runs_to_average):
				# 2 agents: roomba and drone
				# 5 roles: since the scaled team sizes are of size 5
				R = list(range(team_size))
				num_agents = 2
				M = 1
				k_max = 200


				w_min = 1
				w_max = 10
				G = nx.generators.classic.path_graph(2)
				for edge in G.edges:
					G.edges[edge]['weight'] = random.choice(range(w_min, w_max + 1))

				G.add_edge(0, 0, weight=random.choice(range(w_min, w_max + 1)))
				G.add_edge(1, 1, weight=random.choice(range(w_min, w_max + 1)))

				final_sgraph, final_value, sgraphs, values = learn_weighted_synergy_graph(num_agents, R, T, weight_fn_reciprocal, k_max, G=G, display=False)	

				# Get role assignment policy
				p = 0.5
				final_pi, final_value, pis, values = get_approx_optimal_role_assignment_policy(final_sgraph, R, p, weight_fn_reciprocal, k_max)
				roomba_ratios.append(get_roomba_ratio(team_size, final_pi))

			result_str += f"Final graph:\n{final_sgraph}\n"
			result_str += f"pis:{pis}\n"
			result_str += f"values:{values}\n"
			result_str += f"final_pi:{final_pi}\n"
			result_str += f"final value:{final_value}\n"
			result_str += f"avg roomba ratio (out of {runs_to_average} runs):{sum(roomba_ratios) / runs_to_average}\n"
			result_str += f"From data in: {filename}\n\n\n"


		with open(f"avg_results_team_size={team_size}_env={environment_width}_sensorradius={sensor_radius}.txt", "w") as f:
			f.write(result_str)

