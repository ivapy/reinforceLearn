from collections import deque
import numpy as np
import time


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
			while(not self.env.atGoal()):
				#self.env.render()
				action = self.choose_action(observation) # your agent here (this takes random actions)
				oldState = self.env.curr_loc
				observation, reward, done, info = self.env.step(action)
				self.add_to_memory(oldState,observation,reward,action,done)
				self.update(oldState,observation,reward,action,done)

				if done:
					print("iter: " + str(i))
					print action, observation, self.env.goal_loc, done, info

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

















# GAMMA = 0.95
# LEARNING_RATE = 0.001

# MEMORY_SIZE = 1000000
# BATCH_SIZE = 20

# EXPLORATION_MAX = 1.0
# EXPLORATION_MIN = 0.01
# EXPLORATION_DECAY = 0.995

# class rlSeekerDQN:

# 	def __init__(self, observation_space, action_space):
# 		self.exploration_rate = EXPLORATION_MAX

# 		self.action_space = action_space
# 		self.memory = deque(maxlen=MEMORY_SIZE)

# 		self.model = Sequential()
# 		self.model.add(Dense(24, input_shape=(observation_space,), activation="relu"))
# 		self.model.add(Dense(24, activation="relu"))
# 		self.model.add(Dense(self.action_space, activation="linear"))
# 		self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))

# 	def remember(self, state, action, reward, next_state, done):
# 		self.memory.append((state, action, reward, next_state, done))

# 	def act(self, state):
# 		if np.random.rand() < self.exploration_rate:
# 			return random.randrange(self.action_space)
# 		q_values = self.model.predict(state)
# 		return np.argmax(q_values[0])

# 	def experience_replay(self):
# 		if len(self.memory) < BATCH_SIZE:
# 			return
# 		batch = random.sample(self.memory, BATCH_SIZE)
# 		for state, action, reward, state_next, terminal in batch:
# 			q_update = reward
# 			if not terminal:
# 				q_update = (reward + GAMMA * np.amax(self.model.predict(state_next)[0]))
# 			q_values = self.model.predict(state)
# 			q_values[0][action] = q_update
# 			self.model.fit(state, q_values, verbose=0)
# 		self.exploration_rate *= EXPLORATION_DECAY
# 		self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)
