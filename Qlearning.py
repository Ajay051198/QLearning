import numpy as np
import gym

# Creating the environment
env = gym.make("MountainCar-v0")

# Parameters
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 30000
DISP_FREQ = 2000

# Exploration settings
epsilon = 1  # not a constant, qoing to be decayed
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES//2
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

# discritizing the environment space:
DISCRETE_OS_SIZE = [20] *len(env.observation_space.high)
DISCRETE_OS_WIN_SIZE = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE

# initializing the q-table with the appropriate size
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

def get_discrete_state(state):
	discrete_state = (state - env.observation_space.low)/DISCRETE_OS_WIN_SIZE
	return tuple(discrete_state.astype(np.int))

for episode in range(EPISODES):
	if episode % DISP_FREQ == 0:
		render = True
		print(episode)
	else:
		render = False

	# initializing
	discrete_state = get_discrete_state(env.reset())
	done = False

	#updating the qtable
	while not done:

		if np.random.random() > epsilon:
			# Get action from q_table
			action = np.argmax(q_table[discrete_state])
		else:
			# Get a random action
			action = np.random.randint(0, env.action_space.n)

		new_state, reward, done, _ = env.step(action)
		new_discrete_state = get_discrete_state(new_state)

		if render:
			env.render()

		if not done:
			# Maximum possible Q value in next step (for new state)
			max_future_q = np.max(q_table[new_discrete_state])
			# Current Q value (for current state and performed action)
			current_q = q_table[discrete_state + (action,)]
			# Current Q value (for current state and performed action)
			new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
			# # Update Q table with new Q value
			q_table[discrete_state + (action,)] = new_q

		elif new_state[0] >= env.goal_position:
			q_table[discrete_state + (action,)] = 0
			print(f"Reached goal at episode {episode}")

		# update state
		discrete_state = new_discrete_state

		# Decaying is being done every episode if episode number is within decaying range
		if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
			epsilon -= epsilon_decay_value

env.close()
