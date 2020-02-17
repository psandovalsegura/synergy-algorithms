from src.normal_distribution import *

def test_create_normal():
	n = NormalDistribution(0,1)
	assert n.mean == 0
	assert n.variance == 1

def test_add_normals():
	n1 = NormalDistribution(5,3)
	n2 = NormalDistribution(1,1)
	n = n1 + n2

	assert n.mean == 6
	assert n.variance == 4

def test_evaluate():
	n = NormalDistribution(0,1)
	p = 0.95
	e = n.evaluate(p) == norm.ppf(p)