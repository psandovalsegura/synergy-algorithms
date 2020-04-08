import pickle
import random
import itertools
from src.observation_set import *

def create_observation_from_data(oteam_size, sensor_radius, num_completion_times):
	"""
	Create an observation set based on pickled data from Vacuum Simulator experiments

	oteam_size is the team size within the observation set

	Note:
	The original data from Vacuum Simulator is a list of tuples formatted as follows:
	[(sensor_radius, ratio, [completion_time_1, completion_time_2, ...]), ...]
	"""
	# Open old pickle data and create an observation set
	environment_width = 50
	environment_height = 50
	team_size = 30
	runs_to_average = 100
	max_steps = 3000

	filename = "../synergy-algorithms-data/roomba-drone-teams/test11_%s_%s_%s_%s_%s_iter%s.p" % (environment_width, environment_height, team_size, runs_to_average, max_steps, sensor_radius)
	data = pickle.load(open(filename, "rb"))

	# Plot Avg Completion Time vs Roomba Ratio
	filtered_data = list(filter(lambda tup: tup[0] == sensor_radius, data))

	# Extract sensor radius, ratio, and completion time
	r = [tup[1] for tup in filtered_data]
	c = [tup[2] for tup in filtered_data]
	max_completion_time = max([max(ts) for ts in c])

	# Agents 0 - (oteam_size - 1) are roomba
	# Agents oteam_size - (oteam_size * 2) are drone
	roomba_agents = list(range(oteam_size))
	drone_agents = list(range(oteam_size, oteam_size * 2))
	M = 1

	# iterdict = dict()
	teams_considered = []
	ogroups = []

	# 100 completion times were recorded, 
	# so any number under that can scale the size of 
	# the dataset
	for i in range(num_completion_times):
		for tup in filtered_data:
			ratio = tup[1]
			completion_times = tup[2]

			num_roomba = int(ratio * oteam_size)
			num_drone = oteam_size - num_roomba
			
			# if num_roomba not in iterdict.keys()
			# 	iterdict[num_roomba] = 0
			# else:
			# 	iterdict[num_roomba] += 1

			A_roomba = random.sample(roomba_agents, k=num_roomba)
			A_drone = random.sample(drone_agents, k=num_drone)
			A = A_roomba + A_drone

			# Ensure we don't put in a team that's been seen before
			if set(A) in [set(x) for x in teams_considered]:
				continue
			else:
				teams_considered.append(A)

			# A max cleanup time will correspond to 0 performance
			m_completion_times = [[max_completion_time - x] for x in completion_times]
			time = m_completion_times[i]

			ogroup = ObservationGroup(A, M)
			ogroup.add_observations([time])
			ogroups.append(ogroup)

	os = ObservationSet(M, ogroups)	
	output_filename = f"../synergy-algorithms-data/roomba-drone-teams/observation_set_team_size={oteam_size}_env={environment_width}_sensorradius={sensor_radius}_maxtime={max_completion_time}_setsize={os.get_num_groups()}"
	pickle.dump(os, open(output_filename, "wb"))

def main():
	num_completion_times_list = [1,2,4,6,8,10,40,70,100]
	sensor_radius_list = [9,10,11,12,13]
	for sr in sensor_radius_list:
		for nc in num_completion_times_list:
			create_observation_from_data(oteam_size=30, sensor_radius=sr, num_completion_times=nc)

if __name__ == "__main__":
	main()

