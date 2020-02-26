from src.observation_set import *
from src.observations import *

def test_estimate_capability_1():
	M = 1
	mathcal_A = [0,1,2,3]
	k_max = 100

	A = [0,1,2]
	o1 = [[3], [4], [5]]
	observation_group = ObservationGroup(A, M)
	observation_group.add_observations(o1)

	observation_set = ObservationSet(M, [observation_group])
	# todo
	pass