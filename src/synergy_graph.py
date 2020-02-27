import networkx as nx
import matplotlib.pyplot as plt

class SynergyGraph:
	def __init__(self, G, N):
		"""
		G is a networkx graph
		N is a dictionary of node to normal distributions
		"""
		self.graph = G
		self.normal_distributions = N

	def __str__(self):
		keys = [k for k in self.normal_distributions.keys()]
		full_str = "\nSynergyGraph(Graph(nodes:{0}, edges:{1}),\n             Distributions(keys:{2},".format([e for e in self.graph.nodes], [n for n in self.graph.edges], keys)
		for key in keys:
			full_str += "\n                                " + str(self.normal_distributions[key])
		return full_str + "\n"

	def __repr__(self):
		return str(self)

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

	def display(self, nrows, ncols, index, title):
		plt.subplot(nrows, ncols, index, title=title)
		nx.draw(self.graph, with_labels=True, font_weight='bold')

def distance_fn(G, a, b):
	"""
	a and b are nodes in the graph G
	"""
	path = nx.algorithms.bidirectional_shortest_path(G, a, b)
	distance = len(path) - 1
	return distance
	
def display(G):
	"""
	G is a networkx graph
	"""
	plt.subplot(111)
	nx.draw(G, with_labels=True, font_weight='bold')
	plt.show()
