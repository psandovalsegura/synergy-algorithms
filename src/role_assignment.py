import copy
import random
import itertools
import numpy as np
import scipy.optimize
import networkx as nx
import matplotlib.pyplot as plt
from scipy.special import comb
from src.annealing import annealing
from src.weighted_synergy_graph import WeightedSynergyGraph, get_weighted_distance, random_weighted_graph_neighbor
from src.normal_distribution import NormalDistribution

def learn_weighted_synergy_graph(num_agents, R, T, weight_fn, k_max, G=None, display=True):
	"""
	num_agents is the number of agents
	R is a list of roles
	T is a list of training examples [(pi, V(pi))]
	"""
	if G == None:
		G = create_initial_random_weighted_synergy_graph(num_agents)
	C = estimate_capability_by_role(G, R, T, weight_fn)

	# Create initial synergy graph
	initial_wsgraph = WeightedSynergyGraph(G, C)

	value_function = lambda x: log_likelihood_by_role(x, T, weight_fn)
	def random_neighbor(ws):
		G_prime = random_weighted_graph_neighbor(ws.graph)
		C_prime = estimate_capability_by_role(G_prime, R, T, weight_fn)
		return WeightedSynergyGraph(G_prime, C_prime)

	final_sgraph, final_value, sgraphs, values = annealing(initial_wsgraph, value_function, random_neighbor, debug=False, maxsteps=k_max)

	if display:
		# plot num_graphs and the final graph
		num_graphs = 5
		num_steps = len(sgraphs)
		step_size = int(num_steps / num_graphs) if (num_steps / num_graphs) >= 1 else 1
		fig = plt.figure(figsize=(num_graphs * 3, 6))
		for i, sgraph_index in enumerate(range(0, num_steps, step_size)):
			if i <= num_graphs: 
				title = f"Step {sgraph_index} ({values[sgraph_index]:.2f})"
				sgraphs[sgraph_index].display(fig, 2, num_graphs + 1, i + 1, title=title)
		final_title = f"Final {num_steps} ({final_value:.2f})"
		sgraphs[-1].display(fig, 2, num_graphs + 1, num_graphs + 1, title=final_title)
		plt.show()

	return final_sgraph, final_value, sgraphs, values

def create_initial_random_weighted_synergy_graph(num_agents):
	"""
	Create the first ranodm weighted graph for simulated annealing
	"""
	nearest_neighbors = 3
	rewire_prob = 0.30
	w_min = 1
	w_max = 10
	G = nx.generators.random_graphs.connected_watts_strogatz_graph(num_agents, nearest_neighbors, rewire_prob)
	for edge in G.edges:
		G.edges[edge]['weight'] = random.choice(range(w_min, w_max + 1))
	return G

def estimate_capability_by_role(G, R, T, weight_fn):
	"""
	Estimate the means using a least-squares solver
	and returns a dictionary of capabilities

	G is a networkx graph with self-loops and weighted edges
	R is a list of roles 
	T is a list of training examples [(pi, V(pi))]
	"""
	agents = list(G.nodes)
	num_agents = len(agents)
	num_roles = len(R)

	# estimate means
	means = estimate_means_by_role(G, R, T, weight_fn)

	# estimate variances
	# tolerance = 1e-10
	# variance_0 = np.ones(len(num_agents * num_roles))
	# def F(x):
	# 	return -0.5 * np.log(2 * np.pi * x) - 0.5 * ((v - expected_synergy.mean)**2 / (x))

	# variances = scipy.optimize.broyden1(F, variance_0, f_tol=tolerance)

	static_variance = 1

	# Fill in the capabilities dictionary
	C = dict()
	for agent in agents:
		capability_over_roles = []
		for role in R:
			means_index = agent * num_roles + role
			capability_over_roles.append(NormalDistribution(means[means_index], static_variance))
		C[agent] = capability_over_roles

	return C

	
def estimate_means_by_role(G, R, T, weight_fn):
	"""
	Estimate the means using a least-squares solver
	and returns a vector of the means

	G is a networkx graph with self-loops and weighted edges
	R is a list of roles 
	T is a list of training examples [(pi, V(pi))]
	"""
	agents = list(G.nodes)
	num_agents = len(agents)
	num_roles = len(R)
	num_training_examples = len(T)

	M_mean = np.zeros((num_training_examples, num_agents * num_roles))
	b_mean = np.zeros((num_training_examples, 1))

	for i, example in enumerate(T):
		pi, V = example
		b_mean[i] = V
		M_mean[i] = get_means_coefficient_row(G, pi, weight_fn)

	means = np.linalg.lstsq(M_mean, b_mean, rcond=None)[0]
	return np.reshape(means, num_agents * num_roles)

def get_means_coefficient_row(G, pi, weight_fn):
	"""
	Get the row of coefficients which make up the M_mean matrix
	within estimate_means_by_role

	G is a weighted graph
	pi is a role policy
	"""
	agents = list(G.nodes)
	roles = list(pi.keys())
	num_agents = len(agents)
	num_roles = len(roles)
	total_pairs = comb(num_roles, 2, exact=True)
	
	row = np.zeros(num_agents * num_roles)
	for pair in itertools.combinations(roles, r=2):
		r_a, r_b = pair
		a = pi[r_a]
		b = pi[r_b]
		weighted_distance = get_weighted_distance(G, a, b)
		w = weight_fn(weighted_distance)
		agent_index_1 = agents.index(a)
		agent_index_2 = agents.index(b)
		role_index_1 = roles.index(r_a)
		role_index_2 = roles.index(r_b)

		row[agent_index_1 * num_roles + role_index_1] += w
		row[agent_index_2 * num_roles + role_index_2] += w

	scale = (1 / total_pairs)
	return scale * row

def get_approx_optimal_role_assignment_policy(WS, R, p, weight_fn, k_max):
	"""
	Use simulated annealing to find the optimal role policy given

	WS is a WeightedSynergyGraph
	R is a list of roles
	p is the risk factor
	"""
	agents = list(WS.graph.nodes)
	num_agents = len(agents)
	initial_role_policy = create_random_role_assignment(num_agents, R)

	value_function = lambda x: synergy_by_role_policy(WS, x, weight_fn).evaluate(p)
	def random_neighbor(pi):
		pi_prime = random_role_assignment_neighbor(pi, num_agents, R)
		return pi_prime

	final_pi, final_value, pis, values = annealing(initial_role_policy, value_function, random_neighbor, debug=False, maxsteps=k_max)
	return final_pi, final_value, pis, values

def log_likelihood_by_role(WS, T, weight_fn):
	"""
	Compute the log-likelihood of the training data 
	given the WeightedSynergyGraph

	WS is a WeightedSynergyGraph
	T is a list of training examples
	"""
	likelihood = 0
	for example in T:
		pi, V = example
		synergy_by_role = synergy_by_role_policy(WS, pi, weight_fn)
		likelihood += synergy_by_role.logpdf(V).item()
	return likelihood

def synergy_by_role_policy(WS, pi, weight_fn):
	"""
	WS is a WeightedSynergyGraph
	pi is a role policy mapping roles to agents
	"""
	roles = list(pi.keys())
	total_pairs = comb(len(roles), 2, exact=True)

	pair_role_synergies_sum = NormalDistribution(0, 0)
	for pair in itertools.combinations(roles, r=2):
		r_a, r_b = pair
		pair_role_synergies_sum += pairwise_synergy_by_role_policy(WS, weight_fn, pi[r_a], pi[r_b], r_a, r_b)

	scale = (1 / total_pairs)
	return scale * pair_role_synergies_sum

def pairwise_synergy_by_role_policy(WS, weight_fn, a, b, r_a, r_b):
	"""
	Pairwise synergy between two agents a and b
	assigned to roles r_a and r_b in a weighted synergy graph WS
	"""
	distance = WS.get_distance(a, b)
	a_capability_at_r_a = WS.get_capability(a, r_a)
	b_capability_at_r_b = WS.get_capability(b, r_b)
	w = weight_fn(distance)
	return w * (a_capability_at_r_a + b_capability_at_r_b)

def create_random_role_assignment(num_agents, R):
	"""
	num_agents is the number of agents
	R is a list of roles (starting at index 0)
	"""
	pi = dict()
	agents = list(range(num_agents))
	for role in R:
		pi[role] = random.choice(agents)

	return pi

def random_role_assignment_neighbor(pi, num_agents, R):
	"""
	pi is the current role assignment policy
	num_agents is the number of agents
	R is a list of roles (starting at index 0)
	"""
	pi_prime = copy.deepcopy(pi)
	role_to_modify = random.choice(R)

	# Select new agent from set which excludes the previous agent assignment
	agents = list(set(range(num_agents)) - set([pi_prime[role_to_modify]]))
	pi_prime[role_to_modify] = random.choice(agents)
	return pi_prime


