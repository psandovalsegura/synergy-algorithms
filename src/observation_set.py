
class ObservationGroup:
	def __init__(self, A, M):
		self.A = A
		self.M = M
		self.observations = []

	def add_observation(self, observation):
		"""
		observation is a list of the M values
		corresponding to an observed overall 
		performance of members A
		"""
		if len(observation) != self.M:
			raise ValueError('Unable to add observation because it does not have M values')

		self.observations.append(observation)

	def add_observations(self, observations):
		"""
		observations is a list of lists
		"""
		for observation in observations:
			if len(observation) != self.M:
				raise ValueError('Unable to add observations because one does not have M values')

		self.observations += observations

class ObservationSet:
	def __init__(self, M, observation_groups):
		self.observation_groups = observation_groups
		self.M = M

		for observation_group in observation_groups:
			if observation_group.M != M:
				raise ValueError('Unable to create ObservationSet because one group does not have M subtasks')

	def get_num_groups(self):
		return len(self.observation_groups)
		