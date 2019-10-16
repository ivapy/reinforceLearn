"""
RL Grid Seeker environment definition:
An agent in this environment performs actions to try and locate a goal point

"""
import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

class RLSeeker(gym.Env):


	metadata = {
		'render.modes': ['human', 'rgb_array'],
		'video.frames_per_second' : 50
	}

	def __init__(self,curr_loc = np.array([9,9]),
				goal_loc = np.array([3.5,2.5]),
				living_reward = -10,
				goal_reward = 10,
				xLim = np.array([-10,10]),
				yLim = np.array([-10,10]),
				discrete_step = 0.5):


		self.xLim  = xLim
		self.yLim = yLim

		self.goal_reward = goal_reward
		self.living_reward = living_reward
		self.goal_loc = goal_loc
		self.curr_loc = curr_loc
		self.discrete_step = discrete_step
		self.xStates = np.arange(self.xLim[0],self.xLim[1],self.discrete_step)
		self.yStates = np.arange(self.yLim[0],self.yLim[1],self.discrete_step)

		self.action_space = spaces.Discrete(4) #['North', 'South','East','West']
		self.action_list = {'North':np.array([0,discrete_step]),
							'South':np.array([0,-discrete_step]),
							'East':np.array([discrete_step,0]),
							'West':np.array([-discrete_step,0])
							}
		self.observation_space = spaces.Discrete(2)
		self.noise = 0.1
		self.viewer = None



	def step(self, action):
		if action not in self.action_space:
			raise ValueError('Invalid action')
		if action == 0:
			#implement noise here
			self.curr_loc = self.curr_loc + self.action_list['North']
		elif action == 1:
			self.curr_loc = self.curr_loc + self.action_list['South']
		elif action == 2:
			self.curr_loc = self.curr_loc + self.action_list['East']
		elif action == 3:
			self.curr_loc = self.curr_loc + self.action_list['West']
		self.__checkBounds()
		reward = 0;
		if self.__atGoal():
			reward = self.goal_reward
		else:
			reward = self.living_reward

		return self.curr_loc, reward, self.__atGoal(), {}


	def reset(self):
		self.curr_loc[0] = np.random.choice(self.xStates,1)
		self.curr_loc[1] = np.random.choice(self.yStates,1)
		self.goal_loc[0] = np.random.choice(self.xStates,1)
		self.goal_loc[1] = np.random.choice(self.yStates,1)
		return self.curr_loc



	def render(self, mode='human'):
		screen_width = 600
		screen_height = 400

		world_width = (self.xLim[1] - self.xLim[0])*1.25

		scale = screen_width/world_width
		agentRad = 1 * scale
		if self.viewer is None:
			from gym.envs.classic_control import rendering
			self.viewer = rendering.Viewer(screen_width,screen_height)
			agent = rendering.make_circle(radius = agentRad, res = 30)
			self.agenttrans = rendering.Transform()
			agent.add_attr(self.agenttrans)
			agent.set_color(0.6,0,0)
			pos = self.curr_loc



			self.goal = rendering.make_circle(radius = agentRad, res = 30)
			self.goaltrans = rendering.Transform()
			self.goal.add_attr(self.goaltrans)
			self.goal.set_color(0,1,0)

			goalX = self.goal_loc[0]*scale + screen_width / 2
			goalY = self.goal_loc[1]*scale + screen_height / 2
			self.goaltrans.set_translation(goalX, goalY)
			agentX = pos[0]*scale + screen_width / 2
			agentY = pos[1]*scale + screen_height / 2
			self.agenttrans.set_translation(agentX, agentY)
			self.viewer.add_geom(self.goal)
			self.viewer.add_geom(agent)

		pos = self.curr_loc
		goalX = self.goal_loc[0]*scale + screen_width / 2
		goalY = self.goal_loc[1]*scale + screen_height / 2
		self.goaltrans.set_translation(goalX, goalY)
		agentX = pos[0]*scale + screen_width / 2
		agentY = pos[1]*scale + screen_height / 2
		self.agenttrans.set_translation(agentX, agentY)


		return self.viewer.render(return_rgb_array = mode=='rgb_array')


	def close(self):
		if self.viewer:
			self.viewer.close()
			self.viewer = None

	def __checkBounds(self):
		xBoundLower = self.curr_loc[0] < self.xLim[0]
		xBoundUpper = self.curr_loc[0] > self.xLim[1]
		yBoundLower = self.curr_loc[1] < self.yLim[0]
		yBoundUpper = self.curr_loc[1] > self.yLim[1]
		if xBoundUpper:
			self.curr_loc[0] = self.xLim[1]
		elif xBoundLower:
			self.curr_loc[0] = self.xLim[0]

		if yBoundUpper:
			self.curr_loc[1] = self.yLim[1]
		elif yBoundLower:
			self.curr_loc[1] = self.yLim[0]

	def __atGoal(self):
		return min(self.curr_loc == self.goal_loc)







