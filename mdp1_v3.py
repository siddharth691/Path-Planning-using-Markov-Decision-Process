import matplotlib.pyplot as plt
import math
import numpy as np
import random
import base.mdptoolbox as mdptoolbox
import collections
import csv
import time
import pickle
import json

class PathPlanning:
	"""
	Class that builds the model of the world for input to the MDP for path planning
	
	Inputs:
	-------
	maxRow : (int) maximum number of columns in the map
	maxCol : (int) maximum number of columns in the map
	n_obstacle_pts : (int) number of obstacle points
	cor_pr : (float) probability to be assigned to the state transition coresponding to the correct action direction
	wr_pr : (float) probability to be assigned to the state transition corresponding to the all other action directions
			sum of cor_pr and wr_pr for all the actions should be greater equal to 0.99
	n_actions : (int) number of dimensions (directions are assumed to be in anti clockwise direction)
	startRow : (int) row of the starting state
	startCol : (int) column of the starting state
	goalRow : (int) row of the goal state
	goalCol : (int) column of the goal state
	goalReward : (int) reward when next state is goal state
	obstReward : (int) reward when next state is the obstacle 
	stayReward : (int) reward when next state neither goal state nor the obstacle 

	"""

	def __init__(self, maxRow = 50, maxCol = 50, num_obstacle_pts = 50, cor_pr = 0.7, wr_pr = 0.1, n_actions = 4, startRow = 5, startCol = 5, goalRow = 45, goalCol = 45, goalReward = 100, obstReward =-500, stayReward = -5, gamma = 0.9):


		self.maxRow = maxRow
		self.maxCol = maxCol
		self.num_obstacle_pts = num_obstacle_pts
		self.cor_pr = cor_pr
		self.wr_pr = wr_pr
		self.n_actions = n_actions
		self.startRow = startRow
		self.startCol = startCol
		self.goalRow = goalRow
		self.goalCol = goalCol
		self.goalReward = goalReward
		self.obstReward = obstReward
		self.stayReward = stayReward
		self.gamma = gamma

		#Initializing model outputs
		self.not_occupied = 0
		self.oRow = []
		self.oCol = []
		self.m = np.zeros((2, self.maxRow, self.maxCol))
		self.st = np.zeros((self.n_actions, self.maxRow*self.maxCol, self.maxRow*self.maxCol))
		self.num_states = self.maxRow*self.maxCol
		self.rm = np.ones((self.num_states, self.n_actions))*self.stayReward


	def rotate(self, l, n):

		"""
		rotate a list n times in anticlockwise direction
		"""
		return l[n:] + l[:n]

	def get_obstacles(self):
		"""
		returns a list of obstacle state indexes for rows and columns

		"""

		#Getting boundaries

		#Top wall
		row = 0
		for col in range(self.maxCol):
			self.oRow.append(row)
			self.oCol.append(col)

		#Bottom wall
		row = self.maxRow - 1
		for col in range(self.maxCol):
			self.oRow.append(row)
			self.oCol.append(col)

		#Left side wall
		col = 0
		for row in range(1, self.maxRow-1):
			self.oRow.append(row)
			self.oCol.append(col)

		#Right side wall
		col = self.maxCol - 1
		for row in range(1, self.maxRow -1):
			self.oRow.append(row)
			self.oCol.append(col)


		#Randomly selecting obstacle states
		SameRow = 0
		SameCol = 0
		if(self.startRow < self.goalRow):

			RowRange = list(range(0, self.startRow)) + list(range(self.startRow + 1, self.maxRow - 1))

		elif(self.startRow > self.goalRow):

			RowRange = list(range(0, self.goalRow)) + list(range(self.goalRow + 1, self.maxRow - 1))

		else:

			SameRow = 1

		if(self.startCol < self.goalCol):

			ColRange = list(range(0, self.startCol)) + list(range(self.startCol + 1, self.maxCol - 1))

		elif(self.startCol > self.goalCol):

			ColRange = list(range(0, self.goalCol)) + list(range(self.goalCol + 1, self.maxCol - 1))

		else:

			SameCol = 1

			if(SameRow == 1):

				raise ValueError("Start state and goal state are same")


		wallRows = [random.choice(RowRange) for row in range(self.num_obstacle_pts)]
		wallCols = [random.choice(ColRange) for col in range(self.num_obstacle_pts)]

		self.oRow.extend(wallRows)
		self.oCol.extend(wallCols)

		return None


	def build_map(self):

		"""
		builds the map using obstacle state indexes
		"""

		cur_state = 0
		for row in range(self.maxRow):
			for col in range(self.maxCol):
				self.m[0][row][col] = cur_state
				cur_state+=1

		for row, col in zip(self.oRow, self.oCol):
			if(self.not_occupied == 1):

				self.m[1][row][col] = 0
			else:
				self.m[1][row][col] = 1 
				
		return None

	def build_st_trans_matrix(self):
		"""
		Function that builds state transition model for input to the MDP

		"""
		if((self.cor_pr + (self.n_actions - 1)*self.wr_pr) < 0.99):
			raise ValueError ('Sum of probabilities dont match')

		if(self.not_occupied!=0):
			if(self.not_occupied!=1):
				raise ValueError ('not occupied should be either zero or one')

		#Actions should start from right and go in anti clockwise direction

		act_map = []
		cur_states_a1 = []
		for row in range(self.maxRow):
			for col in range(self.maxCol):

				cur_state = self.m[0][row][col]
				cur_occ = self.m[1][row][col]

				if(cur_occ == self.not_occupied):
					
					right_state = self.m[0][row][col + 1]
					right_occ = self.m[1][row][col + 1]

					top_state = self.m[0][row - 1][col]
					top_occ = self.m[1][row - 1][col]

					left_state = self.m[0][row][col - 1]
					left_occ = self.m[1][row][col - 1]

					bottom_state = self.m[0][row + 1][col]
					bottom_occ = self.m[1][row + 1][col]

					action_map = [right_state, top_state, left_state, bottom_state]
					occ_map = [right_occ, top_occ, left_occ, bottom_occ]

					
					for action in range(self.n_actions):

						occ_map_rot = self.rotate(occ_map, action)
						action_map_rot = self.rotate(action_map, action)

						prob_sum = 0
						for inner_action in range(self.n_actions):
							
							#assign the probability of correct prob to the state in the direction of action
							if(inner_action == 0):

								self.st[action][int(cur_state)][int(action_map_rot[inner_action])] = self.cor_pr
								prob_sum+=self.cor_pr

							elif(inner_action !=2):

								self.st[action][int(cur_state)][int(action_map_rot[inner_action])] = self.wr_pr
								prob_sum+=self.wr_pr


		
		for action in range(self.n_actions):

			rowSum = np.sum(self.st[action][:][:], axis = 1)
			zeroInd = np.where(rowSum == 0)[0]
			lessInd = np.where(rowSum < 1)[0]

			#Assigning probability to unreachable states so that sum of each row of st matrix becomes 1	
			for row in zeroInd:
				col = random.choice(range(0, self.num_states -1))
				self.st[action][row][col] = 1

			#If the sum of probability for each row is not one assigning the 1 - total probability to some random state
			for row in lessInd:
				col = random.choice(range(0, self.num_states - 1))
				self.st[action][row][col] += 1 - np.sum(self.st[action][row][:])

		return None


	def build_reward_matrix(self):

		"""
		Function that builds the reward model of the world for input to the MDP for path planning

		"""
		right_action = 0
		top_action = 1
		left_action = 2
		bottom_action = 3

		goal_state = self.m[0][self.goalRow][self.goalCol]
		for row in range(self.maxRow):

			for col in range(self.maxCol):

				cur_occ = self.m[1][row][col]
				if (cur_occ == self.not_occupied):

					cur_state = int(self.m[0][row][col])
					right_state = int(self.m[0][row][col + 1])
					top_state = int(self.m[0][row - 1][col])
					left_state = int(self.m[0][row][col - 1])
					bottom_state = int(self.m[0][row + 1][col])

					right_occ = int(self.m[1][row][col + 1])
					top_occ = int(self.m[1][row - 1][col])
					left_occ = int(self.m[1][row][col - 1])
					bottom_occ = int(self.m[1][row + 1][col])

					if(right_occ != self.not_occupied):
						self.rm[cur_state][right_action] = self.obstReward

					if(right_state == goal_state):
						self.rm[cur_state][right_action] = self.goalReward

					if(top_occ != self.not_occupied):
						self.rm[cur_state][top_action] = self.obstReward

					if(top_state == goal_state):
						self.rm[cur_state][top_action] = self.goalReward

					if(left_occ != self.not_occupied):
						self.rm[cur_state][left_action] = self.obstReward

					if(left_state == goal_state):
						self.rm[cur_state][left_action] = self.goalReward

					if(bottom_occ != self.not_occupied):
						self.rm[cur_state][bottom_action] = self.obstReward

					if(bottom_state == goal_state):
						self.rm[cur_state][bottom_action] = self.goalReward

		return None

def visualize_path(startRow, startCol, goalRow, goalCol, oCol, oRow, num_states, m, optimal_policy, rm,algorithm, eps):

	#Visualize world
	fig, ax = plt.subplots(figsize = (12,12))
	plt.ion()
	ax.scatter(oCol, oRow, marker = 's',s = 700, c = 'black')
	ax.scatter(startCol, startRow, s = 700, c = 'b')
	ax.scatter(goalCol, goalRow, s = 700,c = 'g')
	plt.axis("equal")
	plt.axis('tight')
	title = 'epsilon = ' + str(eps)
	plt.title(title)
	fig.savefig('map1.png')
	

	#Visualize path
	cur_row = startRow
	cur_col = startCol

	opt_row = [startRow]
	opt_col = [startCol]
	max_points = num_states
	cur_point = 0

	# print('current state  row: {}, col: {}'.format(self.startRow, self.startCol))
	while(1):

		cur_state = int(m[0][cur_row][cur_col])
		cur_opt_action = optimal_policy[str(cur_state)]

		if(cur_opt_action == 0):
			cur_row = cur_row
			cur_col = cur_col + 1
		elif(cur_opt_action == 1):
			cur_row = cur_row - 1
			cur_col = cur_col
		elif(cur_opt_action == 2):
			cur_row = cur_row
			cur_col = cur_col - 1
		else:
			cur_row = cur_row + 1
			cur_col = cur_col

		#Printing optimal action
		# print('Action: {}'.format(cur_opt_action))
		# print('Optimal Utility value for this state and this action : {}'.format(self.expected_values[int(cur_state)]))
		# print('Transition probability for current state and current action : {}'.format(self.st[cur_opt_action][int(cur_state)][int(self.m[0][cur_row][cur_col])]))
		# print('Reward for current state to perform current action : {}'.format(self.rm[int(cur_state)][cur_opt_action]))


		opt_row.append(cur_row)
		opt_col.append(cur_col)

		# print('current State row : {}, col : {}'.format(opt_row[-1], opt_col[-1]))

		ax.plot(opt_col, opt_row, linewidth = 5, color = 'red')
		plt.pause(0.1)

		cur_point+=1

		if(cur_row == goalRow):
			if(cur_col == goalCol):
				print('Goal Reached!!')
				break

		if(cur_point == max_points):
			print('Steps limit over!!')
			break



	figname = 'path_'+algorithm +'.png'
	fig.savefig(figname)
	plt.close()

def visualize_policy(maxRow, maxCol, startRow, startCol, goalRow, goalCol, oCol, oRow, num_states, m, optimal_policy, rm, algorithm, eps):

	#Visualize world
	fig, ax = plt.subplots(figsize = (12,12))
	plt.ion()
	ax.scatter(oCol, oRow, marker = 's',s = 700, c = 'black')
	ax.scatter(startCol, startRow, s = 700, c = 'b')
	ax.scatter(goalCol, goalRow, s = 700,c = 'g')
	plt.axis("equal")
	plt.axis('tight')
	title = 'epsilon = ' + str(eps)
	plt.title(title)
	


	#Arrow config
	arrow_head_len = 0.1
	len_arrow= 1-2*arrow_head_len
	arrow = {}
	arrow['right']={'sx':-1*(len_arrow/2), 'sy':0 ,'dx':len_arrow, 'dy':0}
	arrow['top']={'sx':0, 'sy':(len_arrow/2) ,'dx':0, 'dy':-1*len_arrow}
	arrow['left']={'sx': (len_arrow/2), 'sy':0 ,'dx':-1*len_arrow, 'dy':0}
	arrow['bottom']={'sx': 0, 'sy':-1*(len_arrow/2) ,'dx':0, 'dy':len_arrow}


	#Visualize path
	cur_row = startRow
	cur_col = startCol

	opt_row = [startRow]
	opt_col = [startCol]
	max_points = num_states
	cur_point = 0

	# print('current state  row: {}, col: {}'.format(self.startRow, self.startCol))
	for cur_row in range(maxRow):
		for cur_col in range(maxCol):

		
			cur_state = int(m[0][cur_row][cur_col])
			cur_opt_action = optimal_policy[str(cur_state)]



			if(cur_opt_action == 0):
				direction = 'right'
				x = arrow[direction]['sx']+cur_col
				y = arrow[direction]['sy']+cur_row
				dx = arrow[direction]['dx']
				dy = arrow[direction]['dy']

			elif(cur_opt_action == 1):
				direction = 'top'
				x = arrow[direction]['sx']+cur_col
				y = arrow[direction]['sy']+cur_row
				dx = arrow[direction]['dx']
				dy = arrow[direction]['dy']

			elif(cur_opt_action == 2):
				direction = 'left'
				x = arrow[direction]['sx']+cur_col
				y = arrow[direction]['sy']+cur_row
				dx = arrow[direction]['dx']
				dy = arrow[direction]['dy']
			
			else:
				direction = 'bottom'	
				x = arrow[direction]['sx'] + cur_col
				y = arrow[direction]['sy'] + cur_row
				dx = arrow[direction]['dx']
				dy = arrow[direction]['dy']

			ax.arrow(x, y, dx, dy, head_width=0.3, head_length=0.1, fc='k', ec='k')

	figname = 'policy_'+algorithm+'.png'
	fig.savefig(figname)
	plt.close()



def fit_policy(st, rm, gamma, num_states):
	"""
	This function trains an optimal policy using Markov Decision Process using MDPToolbox
	using PolicyIteration

	"""
	data_policy = {}
	data_policy['convergence'] = {}

	for iter in [100, 200, 300, 400, 500, 600, 700, 800, 900]:

		print('Current Iteration: {}'.format(iter))

		data_policy[str(iter)] = {}

		tot_time_start = time.time()
		vi = mdptoolbox.mdp.PolicyIteration(st, rm, gamma, max_iter = 10000000, eval_type = 1)
		# vi.setVerbose()
		time_iter, iter_value, iter_policy, policy_change = vi.run(max_iter = iter)
		tot_time_end = time.time()
		tot_time = tot_time_end - tot_time_start

		policy_change = [int(x) for x in policy_change]
		if(np.any(np.array(iter_value) > iter)):
			raise ValueError('Value loop of Policy Iteration not stopping at maximum iterations provided')


		data_policy[str(iter)]['tot_time'] = tot_time
		data_policy[str(iter)]['time_iter'] = time_iter
		data_policy[str(iter)]['policy_iter'] = iter_policy
		data_policy[str(iter)]['value_iter'] = iter_value
		data_policy[str(iter)]['policy_change'] = policy_change

		optimal_policy = vi.policy
		expected_values = vi.V
		optimal_policy = tuple(int(x) for x in optimal_policy)
		expected_values = tuple(float(x) for x in expected_values)

		optimal_policy = dict(zip(list(range(num_states)), list(optimal_policy)))
		expected_values = list(expected_values)

		data_policy[str(iter)]['optimal_policy'] = optimal_policy
		data_policy[str(iter)]['expected_values'] = expected_values

	print('Convergence')
	tot_time_start = time.time()
	vi = mdptoolbox.mdp.PolicyIteration(st, rm, gamma, max_iter = 10000000, eval_type = 1)
	time_iter, iter_value, iter_policy_policy, policy_change = vi.run(max_iter = 10000)
	tot_time_end = time.time()

	policy_change = [int(x) for x in policy_change]

	optimal_policy = vi.policy
	expected_values = vi.V
	optimal_policy = tuple(int(x) for x in optimal_policy)
	expected_values = tuple(float(x) for x in expected_values)

	optimal_policy = dict(zip(list(range(num_states)), list(optimal_policy)))
	expected_values = list(expected_values)

	data_policy['convergence']['tot_time'] = tot_time_end - tot_time_start
	data_policy['convergence']['time_iter'] = time_iter
	data_policy['convergence']['policy_iter'] = iter_policy_policy
	data_policy['convergence']['value_iter'] = iter_value
	data_policy['convergence']['policy_change'] = policy_change
	data_policy['convergence']['optimal_policy'] = optimal_policy
	data_policy['convergence']['expected_values'] = expected_values

	return data_policy


def store_to_file(data, file_name):

	with open(file_name, 'w') as outfile:
		json.dump(data, outfile)



def fit_value(st, rm, gamma, num_states):
	"""
	This function trains an optimal policy using Markov Decision Process using MDPToolbox
	using ValueIteration

	"""
	data_value = {}
	data_value['convergence'] = {}
	for iter in [100, 200, 300, 400, 500, 600, 700, 800, 900]:

		print('Current Iteration: {}'.format(iter))
		data_value[str(iter)] = {}

		tot_time_start = time.time()
		vi = mdptoolbox.mdp.ValueIteration(st, rm, gamma, max_iter = 10000000, epsilon = 0.0001)
		# vi.setVerbose()
		time_iter, iter_value, variation = vi.run(max_iter = iter)
		tot_time_end = time.time()
		tot_time = tot_time_end - tot_time_start

		if(iter_value > iter):
			raise ValueError('ValueIteration is not stopping at maximum iterations')

		data_value[str(iter)]['tot_time'] = tot_time
		data_value[str(iter)]['time_iter'] = time_iter
		data_value[str(iter)]['value_iter'] = iter_value
		data_value[str(iter)]['variation'] = variation

		optimal_policy = vi.policy
		expected_values = vi.V

		optimal_policy = tuple(int(x) for x in optimal_policy)
		expected_values = tuple(float(x) for x in expected_values)

		optimal_policy = dict(zip(list(range(num_states)), list(optimal_policy)))
		expected_values = list(expected_values)

		data_value[str(iter)]['optimal_policy'] = optimal_policy
		data_value[str(iter)]['expected_values'] = expected_values

	print('Convergence')
	tot_time_start = time.time()
	vi = mdptoolbox.mdp.ValueIteration(st, rm, gamma, max_iter = 10000, epsilon = 0.0001)
	time_iter, iter_value, variation = vi.run(max_iter = 10000)
	tot_time_end = time.time()

	optimal_policy = vi.policy
	expected_values = vi.V

	optimal_policy = tuple(int(x) for x in optimal_policy)
	expected_values = tuple(float(x) for x in expected_values)

	optimal_policy = dict(zip(list(range(num_states)), list(optimal_policy)))
	expected_values = list(expected_values)

	

	data_value['convergence']['tot_time'] = tot_time_end - tot_time_start
	data_value['convergence']['time_iter'] = time_iter
	data_value['convergence']['value_iter'] = iter_value
	data_value['convergence']['variation'] = variation
	data_value['convergence']['optimal_policy'] = optimal_policy
	data_value['convergence']['expected_values'] = expected_values

	return data_value


def get_reward(startRow, startCol, goalRow, goalCol, oCol, oRow, num_states, m, optimal_policy, rm):

	cur_row = startRow
	cur_col = startCol

	opt_row = [startRow]
	opt_col = [startCol]
	max_points = num_states
	cur_point = 0

	# print('current state  row: {}, col: {}'.format(self.startRow, self.startCol))
	total_reward = 0
	while(1):

		try:

			cur_state = int(m[0][cur_row][cur_col])

		except IndexError:

			print('Didnt converge')
			break

		cur_opt_action = optimal_policy[cur_state]
		total_reward += rm[cur_state][cur_opt_action]

		if(cur_opt_action == 0):
			cur_row = cur_row
			cur_col = cur_col + 1
		elif(cur_opt_action == 1):
			cur_row = cur_row - 1
			cur_col = cur_col
		elif(cur_opt_action == 2):
			cur_row = cur_row
			cur_col = cur_col - 1
		else:
			cur_row = cur_row + 1
			cur_col = cur_col

		#Printing optimal action
		# print('Action: {}'.format(cur_opt_action))
		# print('Optimal Utility value for this state and this action : {}'.format(self.expected_values[int(cur_state)]))
		# print('Transition probability for current state and current action : {}'.format(self.st[cur_opt_action][int(cur_state)][int(self.m[0][cur_row][cur_col])]))
		# print('Reward for current state to perform current action : {}'.format(self.rm[int(cur_state)][cur_opt_action]))


		opt_row.append(cur_row)
		opt_col.append(cur_col)

		
		cur_point+=1

		if(cur_row == goalRow):
			if(cur_col == goalCol):
				print('Goal Reached!!')
				break

		if(cur_point == max_points):
			print('Steps limit over!!')
			break

	return total_reward


def fit_Q(st, rm, m, goalRow, goalCol, gamma, num_states, iters):

	with open('data_world.txt') as json_data:
		data_world = json.load(json_data)

	time_total = []
	reward = []
	opt_policy = []
	mean_discrepancy = []
	for epsilon in [0.1, 0.3, 0.5, 0.7, 0.9]:

		print('Current Epsilon: {}'.format(epsilon))
		tot_time_start = time.time()
		ql = mdptoolbox.mdp.QLearning(st, rm, gamma, n_iter = iters)
		policies, rewardi = ql.run(epsilon, 0.3, m[0][goalRow][goalCol], m, decay_eps = 0, decay_alpha = 0)
		tot_time_end = time.time()
		tot_time = tot_time_end - tot_time_start
		time_total.append(tot_time)

		mean_discrepancy.append(rewardi)
		optimal_policy = ql.policy
		optimal_policy = tuple(int(x) for x in optimal_policy)
		optimal_policy = dict(zip([str(x) for x in list(range(num_states))], list(optimal_policy)))
		policies = [tuple(int(x) for x in opt_policy) for opt_policy in policies]
		policies = [dict(zip(list(range(num_states)), list(opt_policy))) for opt_policy in policies]

		reward_policy = []
		for p_pol in policies:

			reward_p = get_reward(data_world['startRow'], data_world['startCol'], data_world['goalRow'], data_world['goalCol'], data_world['oCol'], data_world['oRow'], data_world['num_states'], data_world['m'],p_pol , data_world['rm'])
			reward_policy.append(reward_p)

		reward.append(reward_policy)
		opt_policy.append(optimal_policy)

	return time_total, opt_policy, reward, mean_discrepancy



def plot_analysis(file_data_world, file_data_value, file_data_policy):


	with open(file_data_world) as json_data:
		data_world = json.load(json_data)

	with open(file_data_value) as json_data:
		data_value = json.load(json_data)

	with open(file_data_policy) as json_data:
		data_policy = json.load(json_data)

	print(data_policy['convergence']['policy_iter'])

	tot_time_policy = []
	tot_time_value = []
	for iter in [100, 200, 300, 400, 500, 600, 700, 800, 900]:
		tot_time_policy.append(data_policy[str(iter)]['tot_time'])
		tot_time_value.append(data_value[str(iter)]['tot_time'])

	fig, ax = plt.subplots()
	ax.plot([100, 200, 300, 400, 500, 600, 700, 800, 900], tot_time_policy)
	ax.plot([100, 200, 300, 400, 500, 600, 700, 800, 900], tot_time_value)
	plt.legend(['PolicyIteration', 'ValueIteration'])
	plt.title('Total computation time (s)')
	plt.ylabel('Time in seconds')
	plt.xlabel('Iterations')
	plt.show()

	value_time_policy = []
	value_time_value = []

	for itr in [100, 200, 300, 400, 500, 600, 700, 800, 900]:

		time_policy = []
		for time_value in data_policy[str(itr)]['time_iter']:
			time_policy.append(sum(time_value)/float(len(time_value)))

		value_time_policy.append(sum(time_policy)/len(time_policy))
		value_time_value.append(sum(data_value[str(itr)]['time_iter'])/len(data_value[str(itr)]['time_iter']))


	fig2, ax2 = plt.subplots()
	ax2.plot([100, 200, 300, 400, 500, 600, 700, 800, 900], value_time_policy)
	ax2.plot([100, 200, 300, 400, 500, 600, 700, 800, 900], value_time_value)
	ax2.legend(['PolicyIteration', 'ValueIteration'])
	plt.title('Average time of per value update iteration')
	plt.ylabel('Time in seconds')
	plt.xlabel('Iterations')
	plt.show()


	# # Visualize the path collect the reward for policy iteration, and visualize policy
	# reward_policy = []
	# reward_value = []
	# for file_name in list(data_policy.keys()):
	# 	algorithm = 'PolicyIteration'
	# 	total_reward_policy = visualize_path(data_world['startRow'], data_world['startCol'], data_world['goalRow'], data_world['goalCol'], data_world['oCol'], data_world['oRow'], data_world['num_states'], data_world['m'], data_policy[file_name]['optimal_policy'], data_world['rm'], file_name, algorithm)
	# 	reward_policy.append(total_reward_policy)
	# 	visualize_policy(data_world['maxRow'], data_world['maxCol'], data_world['startRow'], data_world['startCol'], data_world['goalRow'], data_world['goalCol'], data_world['oCol'], data_world['oRow'], data_world['num_states'], data_world['m'], data_policy[file_name]['optimal_policy'], data_world['rm'], file_name, algorithm)

	# 	algorithm = 'ValueIteration'
	# 	total_reward_value = visualize_path(data_world['startRow'], data_world['startCol'], data_world['goalRow'], data_world['goalCol'], data_world['oCol'], data_world['oRow'], data_world['num_states'], data_world['m'], data_value[file_name]['optimal_policy'], data_world['rm'], file_name, algorithm)
	# 	reward_value.append(total_reward_value)
	# 	visualize_policy(data_world['maxRow'], data_world['maxCol'], data_world['startRow'], data_world['startCol'], data_world['goalRow'], data_world['goalCol'], data_world['oCol'], data_world['oRow'], data_world['num_states'], data_world['m'], data_value[file_name]['optimal_policy'], data_world['rm'], file_name, algorithm)
	

	# # Visualize the reward with time for policyIteration and valueIteration
	# print(reward_policy)
	# print(reward_value)
	# fig3, ax3 = plt.subplots()
	# ax3.plot([100, 200, 300, 400, 500, 600, 700, 800, 900], reward_policy[1:])
	# plt.xlabel('Iterations')
	# plt.ylabel('Reward Gathered')
	# plt.title('Policy Iteration')
	# plt.savefig('reward_PolicyIter.png')
	# plt.show()

	# fig4, ax4 = plt.subplots()
	# ax4.plot([100, 200, 300, 400, 500, 600, 700, 800, 900], reward_value[1:])
	# plt.xlabel('Iterations')
	# plt.ylabel('Reward Gathered')
	# plt.title('Value Iteration')
	# plt.savefig('reward_ValueIter.png')
	# plt.show()

def main():

	# world_params = {

	# 'maxRow' : 12,
	# 'maxCol' : 12,
	# 'num_obstacle_pts' : 10,
	# 'cor_pr' : 0.9,
	# 'wr_pr' : 0.05,
	# 'n_actions' : 4,
	# 'startRow' : 2,
	# 'startCol' : 2,
	# 'goalRow' : 10,
	# 'goalCol' : 9,
	# 'goalReward' : 100,
	# 'obstReward' : -500,
	# 'stayReward' : -3,
	# 'gamma' : 0.98
	# }

	# tm = PathPlanning(**world_params)
	# tm.get_obstacles()
	# tm.build_map()
	# tm.build_st_trans_matrix()
	# tm.build_reward_matrix()

	# world_data= {}
	# world_data['st'] = tm.st.tolist()
	# world_data['rm'] = tm.rm.tolist()
	# world_data['gamma'] = tm.gamma
	# world_data['num_states'] = tm.num_states
	# world_data['startRow'] = tm.startRow
	# world_data['startCol'] = tm.startCol
	# world_data['goalRow'] = tm.goalRow
	# world_data['goalCol'] = tm.goalCol
	# world_data['oCol'] = tm.oCol
	# world_data['oRow'] = tm.oRow
	# world_data['m'] = tm.m.tolist()
	# world_data['maxRow'] = tm.maxRow
	# world_data['maxCol'] = tm.maxCol

	# with open('data_world.txt', 'w') as outfile:
	# 	json.dump(world_data, outfile)

	with open('data_world.txt') as json_data:
		world_data = json.load(json_data)
	
	world_data['st'] = np.array(world_data['st'])
	world_data['rm'] = np.array(world_data['rm'])
	world_data['m'] = np.array(world_data['m'])




	##Policy Iteration Experimentations

	# print('Experiments with Policy Iteration')

	# data_policy = fit_policy(world_data['st'], world_data['rm'], world_data['gamma'], world_data['num_states'])

	# #Storing policy data to file

	# print('Storing PolicyIteration data to file')
	# # print(data_policy)

	# store_to_file(data_policy, 'data_policy.txt')

	# ##Value Iteration Experimentations

	# print('Experiments with Value Iteration')
	# data_value = fit_value(world_data['st'], world_data['rm'], world_data['gamma'], world_data['num_states'])

	# #Storing value data to file
	# print('Storing ValueIteration data to file')
	# store_to_file(data_value, 'data_value.txt')


	#Showing plots
	# plot_analysis('data_world.txt', 'data_value.txt', 'data_policy.txt')
	# visualize_path(world_data['startRow'], world_data['startCol'], world_data['goalRow'], world_data['goalCol'], world_data['oCol'], world_data['oRow'], world_data['num_states'], world_data['m'], optimal_policy)
	
	iterations = 5000
	# time_total, opt_policy, reward, mean_discrepancy=  fit_Q(world_data['st'], world_data['rm'], world_data['m'], world_data['goalRow'], world_data['goalCol'], world_data['gamma'], world_data['num_states'], iterations)


	# data_q= {}

	# data_q['time_total'] = time_total
	# data_q['opt_policy'] = opt_policy
	# data_q['reward'] = reward
	# data_q['mean_discrepancy'] = mean_discrepancy

	# with open('data_q.txt', 'w') as outfile:
	# 	json.dump(data_q, outfile)


	

	with open('data_q.txt') as json_data:
		data_q = json.load(json_data)


	time_total = data_q['time_total']
	opt_policy = data_q['opt_policy']
	reward = data_q['reward']
	mean_discrepancy = data_q['mean_discrepancy']


	# #Plot total time with epsilon
	# fig1, ax1 = plt.subplots()
	# plt.plot([0.1, 0.3, 0.5, 0.7, 0.9], time_total)
	# plt.xlabel('Epsilon')
	# plt.ylabel('Total Computation time (s)')
	# plt.title('Q-Learning total time for %d'%iterations)
	# fig1.savefig('ComputationTime-Q.png')


	#Plot reward with iterations for each epsilon

	# fig2, ax2 = plt.subplots()
	# for index, eps in enumerate([0.1, 0.3, 0.5, 0.7, 0.9]):

	# ax2.plot(list(range(1,iterations+1)), reward[-1])

	# plt.xlabel('iterations')
	# plt.ylabel('Reward collected')
	# plt.title('Reward collected varying with iterations (epsilon = 0.9)')
	# plt.legend(['eps = 0.1', 'eps = 0.3', 'eps = 0.5', 'eps = 0.7', 'eps = 0.9'])
	# plt.ylim((-100,100))
	# fig2.savefig('reward_iter_eps-Q.png')

	#Mean discrepancy with iterations for each epsilon

	# fig3, ax3 = plt.subplots()
	# for index, eps in enumerate([0.1, 0.3, 0.5, 0.7, 0.9]):

	# 	ax3.plot(list(range(1, iterations+1)), mean_discrepancy[index])

	# plt.xlabel('iterations')
	# plt.ylabel('Mean of discrepancy (= alpha*delta)')
	# plt.title('Mean discrepancy')
	# plt.legend(['eps = 0.1', 'eps = 0.3', 'eps = 0.5', 'eps = 0.7', 'eps = 0.9'])

	# fig3.savefig('mean_discrepancy-Q.png')



	for index, eps in  enumerate([0.1, 0.3, 0.5, 0.7, 0.9]):

		policy = opt_policy[index]
		algorithm = 'q-learning' + str(eps)
		visualize_policy(world_data['maxRow'], world_data['maxCol'], world_data['startRow'],world_data['startCol'], world_data['goalRow'] , world_data['goalCol'], world_data['oCol'], world_data['oRow'], world_data['num_states'], world_data['m'], policy, world_data['rm'], algorithm, eps)
		visualize_path(world_data['startRow'], world_data['startCol'], world_data['goalRow'], world_data['goalCol'], world_data['oCol'], world_data['oRow'], world_data['num_states'], world_data['m'], policy, world_data['rm'], algorithm, eps)

	# plt.show()

	

if __name__ == '__main__':
	main()



