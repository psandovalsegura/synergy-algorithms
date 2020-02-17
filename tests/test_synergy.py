# import os

# print('current')
# print(os.getcwd())
from src.synergy import weight_fn_reciprocal, weight_fn_exponential

def test_weight_fn_reciprocal():
	assert weight_fn_reciprocal(1) == 1
	assert weight_fn_reciprocal(2) == 0.5

