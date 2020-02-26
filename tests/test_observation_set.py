import pytest
from src.observation_set import *

def test_observation_group_1():
	A = [1,2,3]
	M = 2
	og = ObservationGroup(A, M)
	og.add_observation([50, 51])
	og.add_observations([[52, 53], [54, 55]])
	assert len(og.observations) == 3

def test_observation_group_2():
	A = [1,2,3]
	M = 1
	og = ObservationGroup(A, M)
	og.add_observation([50])
	og.add_observations([[51], [52], [53]])
	assert len(og.observations) == 4

def test_observation_group_3():
	A = [1,2,3]
	M = 1
	with pytest.raises(ValueError) as excinfo:
		og = ObservationGroup(A, M)
		og.add_observation([[51], [52], [53]])
	assert "Unable to add observation" in str(excinfo.value)

def test_observation_group_4():
	A = [1,2,3]
	M = 2
	with pytest.raises(ValueError) as excinfo:
		og = ObservationGroup(A, M)
		og.add_observations([[51], [52, 53]])
	assert "Unable to add observations" in str(excinfo.value)
