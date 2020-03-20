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
		a and b are nodes in the graph G
		"""
		if a == b:
			return self.graph[a][a]['weight']
		else:
			return nx.algorithms.bellman_ford_path_length(self.graph, a, b)

