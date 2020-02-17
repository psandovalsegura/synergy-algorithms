from src.normal_distribution import *

def test_create_normal():
	n = NormalDistribution(0,1)
	assert n.mean == 0
	assert n.variance == 1

def test_add():
	n1 = NormalDistribution(5,3)
	n2 = NormalDistribution(1,1)
	n = n1 + n2

	assert n.mean == 6
	assert n.variance == 4

def test_multiply():
	n1 = 3 * NormalDistribution(5,3)
	assert n1.mean == 15
	assert n1.variance == 27

	n2 = 0.1 * NormalDistribution(1,1) 
	assert n2.mean == 0.1
	assert np.round(n2.variance, 3) == 0.01

def test_add_multiply():
	n = 3 * NormalDistribution(1,2) + 2 * NormalDistribution(3,4)
	assert n.mean == (3 * 1 + 2 * 3)
	assert n.variance == (np.power(3,2) * 2 + np.power(2,2) * 4)

def test_evaluate():
	n = NormalDistribution(0,1)
	p = 0.95
	e = n.evaluate(p) == norm.ppf(p)