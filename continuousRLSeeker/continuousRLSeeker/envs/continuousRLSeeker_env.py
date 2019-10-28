"""
RL Grid Seeker environment definition:
An agent in this environment performs actions to try and locate a goal point

"""
import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

class continuousRLSeeker(gym.Env):


	metadata = {
		'render.modes': ['human', 'rgb_array'],
		'video.frames_per_second' : 50
	}

	def __init__(self,curr_loc = np.array([9,9]),
				goal_reward = 10,
				goal_thresh = 0.1,
				xLim = np.array([-10,10]),
				yLim = np.array([-10,10]),
				discrete_step = 1E-3,
				minVel = -5,
				maxVel = 5,
				minAngle = -math.pi,
				maxAngle = math.pi,
				):
		self.goal_thresh = goal_thresh
		self.xLim = xLim
		self.yLim = yLim
		self.discrete_step = discrete_step
		self.goal = 10
		self.action_space = spaces.Box(low = np.array([minVel, minAngle]),high = np.array([maxVel, maxAngle]), dtype=np.float32)
		self.observation_space = spaces.Box(low = np.array([self.xLim[0], self.yLim[0]]),high = np.array([self.xLim[1], self.yLim[1]]), dtype=np.float32)
		self.curr_loc = np.zeros(2)
		self.goal_loc = np.zeros(2)
		self.curr_loc[0] = (xLim[1] - xLim[0])*np.random.sample() + xLim[0]
		self.curr_loc[1] = (yLim[1] - yLim[0])*np.random.sample() + yLim[0]

		self.goal_loc[0] = (xLim[1] - xLim[0])*np.random.sample() + xLim[0]
		self.goal_loc[1] = (yLim[1] - yLim[0])*np.random.sample() + yLim[0]
		self.viewer = None

	def step(self, action):
		angle = action[1]
		vel = action[0]
		self.curr_loc[0] += vel*math.cos(angle)*self.discrete_step
		self.curr_loc[1] += vel*math.sin(angle)*self.discrete_step
		self.__checkBounds()
		reward = 0;
		if self.atGoal():
			reward = 10
		else:
			r = self.curr_loc - self.goal_loc
			r = np.power(r,2)
			reward = -np.sum(r)

		return self.curr_loc, reward, self.atGoal(), {}


	def reset(self):
		self.curr_loc[0] = (self.xLim[1] - self.xLim[0])*np.random.sample() + self.xLim[0]
		self.curr_loc[1] = (self.yLim[1] - self.yLim[0])*np.random.sample() + self.yLim[0]

		self.goal_loc[0] = (self.xLim[1] - self.xLim[0])*np.random.sample() + self.xLim[0]
		self.goal_loc[1] = (self.yLim[1] - self.yLim[0])*np.random.sample() + self.yLim[0]
		return self.curr_loc



	def render(self, mode='human'):
		outer_width = 800
		outer_height = 800
		screen_width = 600
		screen_height = 600

		world_width = (self.xLim[1] - self.xLim[0])

		scale = screen_width/world_width
		agentRad = 1 * scale
		if self.viewer is None:
			from gym.envs.classic_control import rendering
			self.viewer = rendering.Viewer(outer_width,outer_height)
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

	def atGoal(self):
		return math.sqrt(np.sum(np.power(self.curr_loc - self.goal_loc,2))) < self.goal_thresh







