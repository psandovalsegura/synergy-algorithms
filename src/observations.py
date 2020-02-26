import numpy as np
from src.synergy import synergy

def estimate_capability(O, G, mathcal_A):
	"""
	O is an ObservationSet
	G is a networkx graph
	"""
	pass

def log_likelihood(O, S, weight_fn):
	"""
	O is an ObservationSet
	S is a SynergyGraph
	"""
	likelihood = 0
	for observation_group in O:
		synergy_distributions = synergy(S, observation_group.A, weight_fn)
		for i, observation in enumerate(observation_group.observations):
			# iterate through an observation of each of the M subtasks 
			# and evaluate the observation on the corresponding distribution
			distribution = synergy_distributions[i]
			likelihood += distribution.logpdf(observation)
	return likelihood