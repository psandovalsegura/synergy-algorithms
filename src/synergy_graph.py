import networkx as nx

class SynergyGraph:
	def __init__(self, G, N):
		"""
		G is a networkx graph
		N is a dictionary of node to normal distributions
		"""
		self.graph = G
		self.normal_distributions = N

	def get_distributions(self, a):
		"""
		a is a node in self.graph
		returns list of distributions for a at M subtasks
		"""
		return self.normal_distributions[a]

	def get_distance(self, a, b):
		"""
		a and b are nodes in the graph G
		"""
		return distance_fn(self.graph, a, b)

def distance_fn(G, a, b):
	"""
	a and b are nodes in the graph G
	"""
	path = nx.algorithms.bidirectional_shortest_path(G, a, b)
	distance = len(path) - 1
	return distance
	