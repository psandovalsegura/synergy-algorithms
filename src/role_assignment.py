import copy
import random
import itertools
from scipy.special import comb
from src.weighted_synergy_graph import WeightedSynergyGraph, random_weighted_graph_neighbor
from src.normal_distribution import NormalDistribution

def learn_weighted_synergy_graph(num_agents, R, T, display=True):
	"""
	num_agents is the number of agents
	R is a list of roles
	T is a list of training examples [(pi, V(pi))]
	"""
	nearest_neighbors = 3
	rewire_prob = 0.30
	G = nx.generators.random_graphs.connected_watts_strogatz_graph(num_agents, nearest_neighbors, rewire_prob)
	C = estimate_capability_by_role(G, R, T)

	# Create initial synergy graph
	initial_wsgraph = WeightedSynergyGraph(G, C)
	initial_pi = create_random_role_assignment(num_agents, R)

	value_function = lambda x: log_likelihood_by_role(x, T, weight_fn)
	def random_neighbor(s):
		G_prime = random_weighted_graph_neighbor(s.graph)
		C_prime = estimate_capability_by_role(G_prime, R, T, weight_fn)
		return WeightedSynergyGraph(G_prime, C_prime)

	final_sgraph, final_value, sgraphs, values = annealing(initial_wsgraph, value_function, random_neighbor, debug=False, maxsteps=k_max)

	if display:
		# plot num_graphs and the final graph
		num_graphs = 5
		num_steps = len(sgraphs)
		step_size = int(num_steps / num_graphs) if (num_steps / num_graphs) >= 1 else 1
		for i, sgraph_index in enumerate(range(0, num_steps, step_size)):
			title = f"Step {sgraph_index} ({values[sgraph_index]:.2f})"
			if i <= num_graphs: sgraphs[sgraph_index].display(1, num_graphs + 1, i + 1, title=title)
		final_title = f"Final {num_steps} ({final_value:.2f})"
		sgraphs[-1].display(1, num_graphs + 1, num_graphs + 1, title=final_title)
		plt.show()

	return final_sgraph, final_value, sgraphs, values

def estimate_capability_by_role(G, R, T):
	pass

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


