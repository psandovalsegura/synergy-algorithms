import random
import numpy as np

def acceptance_probability(value, new_value, temperature, debug=True):
	"""
	"""
	if new_value > value:
		if debug: print("    - Acceptance probabilty = 1 as new_value = {} > value = {}...".format(new_value, value))
		return 1
	else:
		prob = np.exp(- (value - new_value) / temperature)
		if debug: print("    - Acceptance probabilty = {:.3g}...".format(prob))
		return prob

def temperature(fraction):
	""" 
	Example of temperature decreasing as the process goes on
	"""
	return max(0.01, min(1, 1 - fraction))

def annealing(random_state,
			  value_function,
			  random_neighbor,
			  acceptance_probability=acceptance_probability,
			  temperature=temperature,
			  maxsteps=1000,
			  debug=True):
	""" 
	Optimize the black-box function 'value_function' with the simulated annealing algorithm
	Source: https://perso.crans.org/besson/publis/notebooks/Simulated_annealing_in_Python.html
	"""
	state = random_state
	value = value_function(state)
	states, values = [state], [value]
	for step in range(maxsteps):
		fraction = step / float(maxsteps)
		T = temperature(fraction)
		new_state = random_neighbor(state)
		new_value = value_function(new_state)
		if debug: print("Step #{:>2}/{:>2} : T = {:>4.3g}, state = {}, value = {:>4.3g}, new_state = {}, new_value = {:>4.3g} ...".format(step, maxsteps, T, state, value, new_state, new_value))
		if acceptance_probability(value, new_value, T, debug=debug) > random.random():
			state, value = new_state, new_value
			states.append(state)
			values.append(value)
	return state, value_function(state), states, values

