from collections import deque

class RLAgent():
	def __init__(discount_factor = 0.99,event_memory = deque(),epsilon = 0.5,learning_rate = 0.001,exploration_decay = 0.999,env):
		self.discount_factor = discount_factor
		self.event_memory = event_memory
		self.epsilon = epsilon
		self.learning_rate = learning_rate
		self.exploration_decay = exploration_decay
		self.env = env



	def update(curr_state,next_state,reward,action,done):
		pass

	def add_to_memory(curr_state,next_state,reward,action,done):
		pass

	def create_agent():
		pass


class DQNAgent(RLAgent):
	def __init__(discount_factor,event_memory = ,epsilon,learning_rate,exploration_decay,env):
		super(discount_factor,event_memory,epsilon,learning_rate,exploration_decay,env)
















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
