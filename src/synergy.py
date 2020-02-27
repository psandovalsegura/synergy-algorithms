import itertools
import functools
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.special import comb
from scipy.optimize import basinhopping
from src.annealing import annealing
from src.observations import estimate_capability
from src.synergy_graph import SynergyGraph

def create_synergy_graph(O, mathcal_A, weight_fn, k_max, display=False):
	"""
	O is an observation set
	mathcal_A is the set of all agents
	note: agents should start from 0 since the watts strogatz 
	is labeled that way
	"""
	num_agents = len(mathcal_A)
	nearest_neighbors = 3
	rewire_prob = 0.30
	G = nx.generators.random_graphs.connected_watts_strogatz_graph(num_agents, nearest_neighbors, rewire_prob)
	N = estimate_capability(O, G, weight_fn)

	# Create initial synergy graph
	initial_sgraph = SynergyGraph(G, N)

	value_function = lambda x: log_likelihood(O, x, weight_fn)
	def random_neighbor(s):
		G_prime = random_graph_neighbor(s.graph)
		N_prime = estimate_capability(O, G_prime, weight_fn)
		return SynergyGraph(G_prime, N_prime)

	final_sgraph, final_value, sgraphs, values = annealing(initial_sgraph, value_function, random_neighbor, debug=True, maxsteps=k_max)

	if display:
		# plot num_graphs and the final graph
		num_graphs = 4
		for i, sgraph_index in enumerate(range(0, k_max, int(k_max / 4))):
			title = f"Iter {sgraph_index} ({values[sgraph_index]:.2f})"
			sgraphs[sgraph_index].display(1, num_graphs + 1, i+1, title=title)
		final_title = f"Final ({final_value:.2f})"
		sgraphs[-1].display(1, num_graphs + 1, num_graphs + 1, title=final_title)
		plt.show()

	return final_sgraph, final_value, sgraphs, values

def log_likelihood(O, S, weight_fn):
	"""
	O is an ObservationSet
	S is a SynergyGraph
	"""
	likelihood = 0
	for observation_group in O.observation_groups:
		synergy_distributions = synergy(S, observation_group.A, weight_fn)
		for observation in observation_group.observations:
			for m, value in enumerate(observation):
				# iterate through an observation of each of the M subtasks 
				# and evaluate the observation on the corresponding distribution
				distribution = synergy_distributions[m]
				likelihood += distribution.logpdf(value).item()
	return likelihood

def random_graph_neighbor(G):
	"""
	G is a networkx graph

	either adds a new random edge or removes
	an existing edge, subject to the constraint 
	that G remains connected
	note: modifies the input parameter G
	"""
	edges = [e for e in G.edges]
	nodes = [n for n in G]
	while True:
		if random.random() < 0.5:
			# if there are no edges to remove, do nothing
			if len(edges) == 0:
				break
			removal_edge = random.choice(edges)
			G.remove_edge(*removal_edge)
		else:
			potential_new_edges = set(itertools.combinations(nodes, r=2)) - set(edges)
			# if there are no new edges to add, do nothing
			if len(potential_new_edges) == 0:
				break
			new_edge = random.choice(list(potential_new_edges))
			G.add_edge(*new_edge)

		# if the graph becomes disconnected we
		# should add back the edge which was removed
		if nx.is_connected(G):
			break
		else:
			G.add_edge(*removal_edge)

	return G

def get_approx_optimal_team_brute(S, mathcal_A, p, k_max, weight_fn):
	"""
	S is a SynergyGraph
	mathcal_A is the set of all agents
	p is the risk factor
	"""
	num_agents = len(mathcal_A)
	best_value = -1
	best_team = None
	for n in range(2, num_agents + 1):
		team, value, _, _ = get_approx_optimal_team(S, mathcal_A, n, p, k_max, weight_fn)
		if value > best_value:
			best_value = value
			best_team = team
	return best_team, best_value

def get_approx_optimal_team(S, mathcal_A, n, p, k_max, weight_fn):
	"""
	S is a SynergyGraph
	mathcal_A is the set of all agents
	n is the optimal size of the team, if unknown use brute force version
	p is the risk factor
	"""
	initial_team = random.sample(mathcal_A, n)
	value_function = lambda x: value_fn_sum(synergy(S, x, weight_fn), p)
	random_neighbor = lambda a: random_team_neighbor(mathcal_A, a)

	final_team, final_value, teams, values = annealing(initial_team, value_function, random_neighbor, debug=True, maxsteps=k_max)
	return final_team, final_value, teams, values

def synergy(S, A, weight_fn):
	"""
	S is a SynergyGraph
	A is a list of agents
	Note: A must be of size >= 2
	"""
	total_pairs = comb(len(A), 2, exact=True)

	pair_synergies = []
	for pair in itertools.combinations(A, r=2):
		pair_synergies.append(pairwise_synergy(S, weight_fn, *pair))

	synergy = functools.reduce(lambda a_distributions, b_distributions : elementwise_add(a_distributions, b_distributions), pair_synergies)
	scale = (1 / total_pairs)
	return list(map(lambda d: scale * d, synergy))

def pairwise_synergy(S, weight_fn, a, b):
	"""
	Pairwise synergy between two agents a and b
	in a synergy graph S
	"""
	distance = S.get_distance(a, b)
	a_distribution = S.get_distributions(a)
	b_distribution = S.get_distributions(b)
	sum_distribution = elementwise_add(a_distribution, b_distribution)
	w = weight_fn(distance)
	return list(map(lambda d: w * d, sum_distribution))

def random_team_neighbor(mathcal_A, A):
	"""
	mathcal_A is the set of all agents
	A is a list of agents

	note: this function does not support 
	multiple agents of the same type
	"""
	if len(mathcal_A) == 0:
		raise RuntimeError('There are no agents available, mathcal_A is empty')

	mathcal_A_set = set(mathcal_A)
	A_set = set(A)
	difference = mathcal_A_set - A_set

	# if A contains all the available agents
	# then nothing can be done
	if len(difference) == 0:
		return A

	new_agent = random.choice(list(difference))
	random_index = random.choice(range(len(A)))
	B = A.copy()
	B[random_index] = new_agent
	return B

def weight_fn_reciprocal(d):
	return 1 / d

def weight_fn_exponential(d, h):
	return np.exp(d * np.log(2) / h)

def value_fn_sum(distributions, p):
	return sum(list(map(lambda distr: distr.evaluate(p), distributions)))

def elementwise_add(a_distributions, b_distributions):
	assert len(a_distributions) == len(b_distributions)
	result = []
	for i in range(len(a_distributions)):
		result.append(a_distributions[i] + b_distributions[i])
	return result