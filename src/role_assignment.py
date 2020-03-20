import itertools
from scipy.special import comb
from src.weighted_synergy_graph import WeightedSynergyGraph
from src.normal_distribution import NormalDistribution

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