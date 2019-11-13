from collections import deque
import numpy as np
import time
import tensorflow as tf
from tensorflow.python import tf2
if not tf2.enabled():
  import tensorflow.compat.v2 as tf
  tf.enable_v2_behavior()
  assert tf2.enabled()

import tensorflow_probability as tfp
from tensorflow import keras
from keras import backend as K
from keras.models import Model
from keras.layers import Layer, Dense, Input, Concatenate, Flatten, Add
from keras.optimizers import Adam
from keras import initializers
import random

class QLearningAgent():
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
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#


class DeepQAgent():
		def __init__(self,env,discount_factor = 0.99,event_memory = deque(),epsilon = 0.5,learning_rate = 0.001,exploration_decay = 1,mem_size = 1E3, max_eps = int(1E3)):
		self.discount_factor = discount_factor
		self.event_memory = event_memory
		self.epsilon = epsilon
		self.learning_rate = learning_rate
		self.exploration_decay = exploration_decay
		self.max_memory = mem_size
		self.max_eps = max_eps
		self.env = env
		self.batch_size = 50
		self.create_agent()


	def update(self):
		if len(event_memory) < self.batch_size:
			return

		minibatch = random.sample(self.event_memory, batch_size)

		for state, next_state, reward, action, terminated in minibatch:

			target = self.policy.predict(state)

			if terminated:
				target[0][action] = reward
			else:
				t = self.policy.predict(next_state)
				target[0][action] = reward + self.gamma * np.argmax(t)

			self.policy.fit(state, target, epochs=1, verbose=0)

	def add_to_memory(self,curr_state,next_state,reward,action,done):
		if len(self.event_memory) >= self.max_memory:
			self.event_memory.popleft()
		self.event_memory.append((curr_state,next_state,reward,action,done))

	def create_agent(self):
		state_input = Input(shape=self.env.observation_space.shape)
		delta = Input(shape = [1])
		h1 = Dense(24, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros')(state_input)
		h2 = Dense(48, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros')(h1)
		h3 = Dense(24, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros')(h2)
		output = Dense(self.env.action_space.shape[0], activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros')(h3)
		self.policy = Model(input = [state_input], output = [output])

		return self.policy

	def choose_action(self,state):
		s = tuple(state)
		options = self.policy.predict(state)
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

#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#


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

#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#


class ActorCritic():

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
		self.sess = tf.Session()

	def __create_actor(self):
		state_input = Input(shape=self.env.observation_space.shape)
		delta = Input(shape = [1])
		h1 = Dense(24, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros')(state_input)
		h2 = Dense(48, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros')(h1)
		h3 = Dense(24, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros')(h2)
		vel_output = Dense(2, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros')(h3)
		ang_output = Dense(2, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros')(h3)
		vel_model = Model(input=[state_input, delta], output=vel_output)
		ang_model = Model(input=[state_input, delta], output=ang_output)

		vel_action_placeholder = tf.placeholder(tf.float32)
		ang_action_placeholder = tf.placeholder(tf.float32)
		vel_delta_placeholder = tf.placeholder(tf.float32)
		ang_delta_placeholder = tf.placeholder(tf.float32)
		vel_norm_dist = tf.placeholder(tfd.Normal(loc = 0,scale = 1))
		ang_norm_dist = tf.placeholder(tfd.Normal(loc = 0,scale = 1))


		self.loss_actor = -tf.log(ang_norm_dist.prob(ang_action_placeholder)*vel_norm_dist.prob(vel_action_placeholder) + 1e-5) * delta_placeholder
		self.training_op_actor = tf.train.AdamOptimizer(lr_actor, name='actor_optimizer').minimize(self.loss_actor)
		return model




	def __create_critic(self):
		state_input = Input(shape = (2,), name = 'State_In')
		action_input = Input(shape = (2,), name = 'Action_In')

		sa_concat = Concatenate([state_input,action_input])
		quadLayer = LinQuadLayer()(sa_concat)
		fc_linear = Dense(1)(sa_concat)
		fc_quad = Dense(1)(quadLayer)
		addLayer = Add()([fc_linear, fc_quad])

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


	def __train_actor(self, action ,advantage, cur_state, vel_dist, ang_dist):
		self.sess.run([self.training_op_actor, self.loss_actor], feed_dict = {norm_dist: action_placeholder: action, delta_placeholder: advantage})

	def train(self):
		pass
	def chooseAction(self, state):
		s = tuple(state)
		op = self.actor.predict(state)
		op = np.array(op)
		mu_vel = op[0]
		sigma_vel = op[1]
		mu_ang = op[2]
		sigma_ang = op[3]
		vel_dist = tfp.distributions.Normal(loc = mu_vel, scale = sigma_vel)
		ang_dist = tfp.distributions.Normal(loc = mu_ang, scale = sigma_ang)
		op_vel = tf.clip_by_value(vel_dist.sample(),self.env.action_space.low[0],self.env.action_space.high[0])
		op_ang = tf.clip_by_value(ang_dist.sample(),self.env.action_space.low[1],self.env.action_space.high[1])
		return op_vel, op_ang, vel_dist, ang_dist


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




































"""

class A2CAgent:
	def __init__(self, state_size, action_size):
		# if you want to see Cartpole learning, then change to True
		self.render = False
		self.load_model = False
		self.state_size = state_size
		self.action_size = action_size
		self.value_size = 1

		# get gym environment name
		# these are hyper parameters for the A3C
		self.actor_lr = 0.0001
		self.critic_lr = 0.001
		self.discount_factor = .9
		self.hidden1, self.hidden2 = 24, 24

		# create model for actor and critic network
		self.actor, self.critic = self.build_model()

		# method for training actor and critic network
		self.optimizer = [self.actor_optimizer(), self.critic_optimizer()]

		if self.load_model:
			self.actor.load_weights("./save_model/cartpole_actor.h5")
			self.critic.load_weights("./save_model/cartpole_critic.h5")

	def build_model(self):
		state = Input(batch_shape=(None, self.state_size))
		actor_input = Dense(30, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform')(state)
		# actor_hidden = Dense(self.hidden2, activation='relu')(actor_input)
		mu_0 = Dense(self.action_size, activation='tanh', kernel_initializer='he_uniform')(actor_input)
		sigma_0 = Dense(self.action_size, activation='softplus', kernel_initializer='he_uniform')(actor_input)

		mu = Lambda(lambda x: x * 2)(mu_0)
		sigma = Lambda(lambda x: x + 0.0001)(sigma_0)


		critic_input = Input(shape = (2,), name = 'State_In')
		action_input = Input(shape = (2,), name = 'Action_In')

		sa_concat = Concatenate([state_input,action_input])
		quadLayer = LinQuadLayer()(sa_concat)
		fc_linear = Dense(1)(sa_concat)
		fc_quad = Dense(1)(quadLayer)
		addLayer = Add()([fc_linear, fc_quad])

		reward_estimator = Model(inputs = [state_input, action_input], outputs = addLayer)


		actor = Model(inputs=state, outputs=(mu, sigma))
		critic = Model(inputs=critic_input, outputs=reward_estimator)

		actor._make_predict_function()
		critic._make_predict_function()

		actor.summary()
		critic.summary()

		return actor, critic

	def actor_optimizer(self):
		action = K.placeholder(shape=(None, 1))
		advantages = K.placeholder(shape=(None, 1))

		# mu = K.placeholder(shape=(None, self.action_size))
		# sigma_sq = K.placeholder(shape=(None, self.action_size))

		mu, sigma_sq = self.actor.output

		pdf = 1. / K.sqrt(2. * np.pi * sigma_sq) * K.exp(-K.square(action - mu) / (2. * sigma_sq))
		log_pdf = K.log(pdf + K.epsilon())
		entropy = K.sum(0.5 * (K.log(2. * np.pi * sigma_sq) + 1.))

		exp_v = log_pdf * advantages

		exp_v = K.sum(exp_v + 0.01 * entropy)
		actor_loss = -exp_v

		optimizer = Adam(lr=self.actor_lr)
		updates = optimizer.get_updates(self.actor.trainable_weights, [], actor_loss)

		train = K.function([self.actor.input, action, advantages], [], updates=updates)
		return train

	# make loss function for Value approximation
	def critic_optimizer(self):
		discounted_reward = K.placeholder(shape=(None, 1))

		value = self.critic.output

		loss = K.mean(K.square(discounted_reward - value))

		optimizer = Adam(lr=self.critic_lr)
		updates = optimizer.get_updates(self.critic.trainable_weights, [], loss)
		train = K.function([self.critic.input, discounted_reward], [], updates=updates)
		return train

	# using the output of policy network, pick action stochastically
	def get_action(self, state):
		mu, sigma_sq = self.actor.predict(np.reshape(state, [1, self.state_size]))
		# sigma_sq = np.log(np.exp(sigma_sq + 1))
		epsilon = np.random.randn(self.action_size)
		# action = norm.rvs(loc=mu, scale=sigma_sq,size=1)
		action = mu + np.sqrt(sigma_sq) * epsilon
		action = np.clip(action, -2, 2)
		return action

	# update policy network every episode
	def train_model(self, state, action, reward, next_state, done):
		target = np.zeros((1, self.value_size))
		advantages = np.zeros((1, self.action_size))

		value = self.critic.predict(state)[0]
		next_value = self.critic.predict(next_state)[0]

		if done:
			advantages[0] = reward - value
			target[0][0] = reward
		else:
			advantages[0] = reward + self.discount_factor * (next_value) - value
			target[0][0] = reward + self.discount_factor * next_value

		self.optimizer[0]([state, action, advantages])
		self.optimizer[1]([[state, action], target])
"""
