from collections import deque
import numpy as np
import time
from tensorflow import keras
import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers import Layer, Dense, Input
from keras.optimizers import Adam



class DiscreteRLAgent():
	def __init__(self,env,discount_factor = 0.99,event_memory = deque(),epsilon = 0.5,learning_rate = 0.001,exploration_decay = 0.999,mem_size = 1E3, max_eps = 1000):
		self.discount_factor = discount_factor
		self.event_memory = event_memory
		self.epsilon = epsilon
		self.learning_rate = learning_rate
		self.exploration_decay = exploration_decay
		self.max_memory = mem_size
		self.max_eps = max_eps
		self.env = env
		self.model = self.create_agent()



	def update(curr_state,next_state,reward,action,done):
		pass

	def add_to_memory(curr_state,next_state,reward,action,done):
		pass

	def create_agent():
		pass

	def choose_action(state):
		pass

	def train():
		pass

	def testPolicy():
		pass


class QLearningAgent(DiscreteRLAgent):
	def __init__(self,env,discount_factor = 0.99,event_memory = deque(),epsilon = 0.5,learning_rate = 0.001,exploration_decay = 1,mem_size = 1E3, max_eps = int(1E3)):
		self.discount_factor = discount_factor
		self.event_memory = event_memory
		self.epsilon = epsilon
		self.learning_rate = learning_rate
		self.exploration_decay = exploration_decay
		self.max_memory = mem_size
		self.max_eps = max_eps
		self.env = env
		self.create_agent()


	def update(self,curr_state,next_state,reward,action,done):
		if done:
			return
		else:
			next_action = self.choose_action(next_state)
			cs = tuple(curr_state)
			ns = tuple(next_state)
			self.Q[cs][action] = (1 - self.learning_rate)*(self.Q[cs][action]) + self.learning_rate*(reward + self.discount_factor*self.Q[ns][next_action])

	def add_to_memory(self,curr_state,next_state,reward,action,done):
		if len(self.event_memory) >= self.max_memory:
			self.event_memory.popleft()
		self.event_memory.append((curr_state,next_state,reward,action,done))

	def create_agent(self):
		self.Q = dict()
		for s1 in range(self.env.xLim[0],self.env.xLim[1] + 1):
			for s2 in range(self.env.yLim[0],self.env.yLim[1] + 1):
				self.Q[(s1,s2)] = np.random.rand(self.env.action_space.n)

	def choose_action(self,state):
		s = tuple(state)
		options = self.Q[s]
		if np.random.uniform() > self.epsilon:
			return np.argmax(options)
		else:
			self.epsilon *= self.exploration_decay
			return np.random.choice(np.array(range(self.env.action_space.n)))



	def train(self):
		for i in range(self.max_eps):
			observation = self.env.reset()
			numIters = 0
			while(not self.env.atGoal()):

				#self.env.render()
				action = self.choose_action(observation) # your agent here (this takes random actions)
				oldState = self.env.curr_loc
				observation, reward, done, info = self.env.step(action)
				self.add_to_memory(oldState,observation,reward,action,done)
				self.update(oldState,observation,reward,action,done)
				numIters += 1
				if done:

					print("iter: " + str(i))
					print action, reward, observation, self.env.goal_loc, done, numIters
					self.env.reset()
					numIters = 0

	def testPolicy(self):
		for _ in range(self.max_eps):
			observation = self.env.reset()
			while(not self.env.atGoal()):
				self.env.render()
				action = self.choose_action(observation) # your agent here (this takes random actions)
				oldState = self.env.curr_loc
				observation, reward, done, info = self.env.step(action)
				self.add_to_memory()
				self.update(oldState,observation,reward,action,done)
				print action, observation, self.env.goal_loc, done, info
				if done:
					observation = self.env.reset()
				time.sleep(0.01)

class LinQuadLayer(layers.Layer):
	def __init__(self):
		super(LinQuadLayer,self).__init__()
		self.numOutputs = 1

	def call(self,inputs):
		if inputs.shape[0] < 1:
			raise ValueError("Cannot have layer input size less than 1")
		u,v = tf.meshgrid(inputs,inputs)
		quadLayer = u * v;
		r,c = quadLayer.shape
		quadLayer = (u * v).numpy();
		op = np.zeros([self.__triangle(inputs.shape[0])])
		for i in range(r):
			for j in range(i):
				op[i * inputs.shape[0] + j] = quadLayer[i][j]

		return tf.convert_to_tensor(op)

	def __triangle(self,N):
		return (N**2 + N)/2


class ContinuousRLAgent():
	def __init__(self,env,discount_factor = 0.99,event_memory = deque(),epsilon = 0.5,learning_rate = 0.001,exploration_decay = 0.999,mem_size = 1E3, max_eps = 1000):
		self.discount_factor = discount_factor
		self.event_memory = event_memory
		self.epsilon = epsilon
		self.learning_rate = learning_rate
		self.exploration_decay = exploration_decay
		self.max_memory = mem_size
		self.max_eps = max_eps
		self.env = env
		self.model = self.create_agent()



	def update(curr_state,next_state,reward,action,done):
		pass

	def add_to_memory(curr_state,next_state,reward,action,done):
		pass

	def create_agent():
		pass

	def choose_action(state):
		pass

	def train():
		pass

	def testPolicy():
		pass


class ActorCritic(ContinuousRLAgent):

	def __init__(self,env,discount_factor = 0.99,event_memory = deque(),mem_size = 2E3, max_eps = 1000,batch_size = 50):
		self.max_eps = max_eps
		self.event_memory = event_memory
		self.discount_factor = discount_factor
		self.mem_size = mem_size
		self.batch_size = batch_size
		self.critic = self.__create_critic()
		self.actor = self.__create_actor()
		self.alpha = 0.0001
		self.beta = 0.0005

	def __create_actor(self):
		state_input = Input(shape=self.env.observation_space.shape)
		delta = Input(shape = [1])
		h1 = Dense(24, activation='relu')(state_input)
		h2 = Dense(48, activation='relu')(h1)
		h3 = Dense(24, activation='relu')(h2)
		output = Dense(self.env.action_space.shape[0], activation='softmax')(h3)

		model = Model(input=[state_input, delta], output=output)
		adam  = Adam(lr=0.001)
		model.compile(loss="mse", optimizer=adam)

		policy = Model(input = [state_input], output = [output])
		return model, policy


	def __log_loss(y_true,y_pred):
		pass


	def __create_critic(self):
		state_input = Input(shape = (2,), name = 'State_In')
		action_input = Input(shape = (2,), name = 'Action_In')

		sa_concat = layers.concatenate([state_input,action_input])
		quadLayer = LinQuadLayer()(sa_concat)
		fc_linear = layers.Dense(1)(sa_concat)
		fc_quad = layers.Dense(1)(quadLayer)
		addLayer = layers.Add()([fc_linear, fc_quad])

		reward_estimator = Model(inputs = [state_input, action_input], outputs = addLayer)
		adam  = Adam(lr=0.001)
		reward_estimator.compile(loss="mse", optimizer=adam)
		return reward_estimator

	def __train_critic(self):
		samples = random.sample(self.event_memory, self.batch_size)
		for sample in samples:
			cur_state, action, reward, new_state, done = sample
				if not done:
					target_action = self.actor.predict(new_state)
					future_reward = self.critic.predict([new_state target_action])

					reward += self.discount_factor * future_reward - reward
				self.critic.fit([cur_state, action], reward, verbose=0)


	def __train_actor(self):
		pass

	def train(self):
		pass

	def update(self):
		pass

	def add_to_memory(self, state):
		if len(self.event_memory) > self.mem_size:
			self.event_memory.pop()
			self.event_memory.appendleft(state)
		else:
			self.event_memory.appendleft(state)

	def testPolicy(self):
		pass


















