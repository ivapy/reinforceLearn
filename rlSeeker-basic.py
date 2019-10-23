
import rlSeeker
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import time
import Agents
from Agents import QLearningAgent


ENV_NAME = "rlSeeker-v0"
MAX_EPS = 1000


# class QLearningAgent:

# def main():
# 	env = gym.make(ENV_NAME)
# 	observation_space = env.observation_space.n
# 	print("____________________________")
# 	print(env.observation_space.n)
# 	action_space = env.action_space.n
# 	dqn_solver = DQNSolver(observation_space, action_space)
# 	run = 0
# 	while True:
# 		run += 1
# 		state = env.reset()
# 		state = np.reshape(state, [1, observation_space])
# 		step = 0
# 		while True:
# 			step += 1
# 			env.render()
# 			action = dqn_solver.act(state)
# 			state_next, reward, terminal, info = env.step(action)
# 			reward = reward if not terminal else -reward
# 			state_next = np.reshape(state_next, [1, observation_space])
# 			dqn_solver.remember(state, action, reward, state_next, terminal)
# 			state = state_next
# 			if terminal:
# 				print "Run: " + str(run) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(step)
# 				break
# 			dqn_solver.experience_replay()


def main():

	environment = gym.make("rlSeeker-v0")
	agent = QLearningAgent(env = environment)
	agent.train()




	env.close()

if __name__ == "__main__":
	main()

