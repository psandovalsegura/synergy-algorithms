import copy
import random
import networkx as nx
from src.synergy_graph import SynergyGraph

class WeightedSynergyGraph(SynergyGraph):
	def __init__(self, G, N):
		"""
		G is a networkx graph with self-loops and weighted edges
		N is a dictionary of node to normal distributions

		Note: Here, N takes an agent node k and maps to a list
		of normals representing agent k's performance on each of 
		the M roles
		"""
		super().__init__(G, N)

	def get_capability(self, a, r_a):
		"""
		a is a node in self.graph
		returns list of distributions for a at M subtasks

		Note: roles must be indexed starting at 0
		"""
		return self.normal_distributions[a][r_a]

	def get_distance(self, a, b):
		"""
		a and b are nodes in the weighted graph G
		"""
		return get_weighted_distance(self.graph, a, b)

def get_weighted_distance(G, a, b):
	"""
	a and b are nodes in the weighted graph G
	"""
	if a == b:
		return G[a][a]['weight']
	else:
		return nx.algorithms.bellman_ford_path_length(G, a, b)

def random_weighted_graph_neighbor(G):
	"""
	G is a networkx graph with self-loops and weighted edges

	either increases the weight of a random edge by 1 (subject to a max of w_max), 
	or decreases the weight of a random edge by 1 (subject to a min of w_min),
	or removes an existing edge, subject to the constraint that G remains connected,
	or adds a new random edge
	"""
	H = copy.deepcopy(G)
	edges = [e for e in H.edges]
	nodes = [n for n in H]

	while True:
		rand = random.random()
		if rand < 0.25:
			# increase weight of random edge by 1
			increase_edge = random.choice(edges)
			H[increase_edge[0]][increase_edge[1]]['weight'] += 1
		elif rand < 0.5:
			# decrease weight of random edge by 1
			decrease_edge = random.choice(edges)
			H[decrease_edge[0]][decrease_edge[1]]['weight'] -= 1
		elif rand < 0.75:
			# if there are no edges to remove, do nothing
			if len(edges) == 0:
				break
			removal_edge = random.choice(edges)
			H.remove_edge(*removal_edge)
		else:
			potential_new_edges = set(itertools.combinations(nodes, r=2)) - set(edges)
			# if there are no new edges to add, do nothing
			if len(potential_new_edges) == 0:
				break
			new_edge = random.choice(list(potential_new_edges))
			H.add_edge(*new_edge)

		# if the graph becomes disconnected we
		# should add back the edge which was removed
		if nx.is_connected(H):
			break
		else:
			H.add_edge(*removal_edge)

	return H
