import numpy as np
import gym
import q_learning

# create environement
env = gym.make("Taxi-v3")
env.render()

# create Q-table and initialize it
action_size = env.action_space.n
print("Action size ", action_size)

state_size = env.observation_space.n
print("State size ", state_size)

qtable = np.zeros((state_size, action_size))
print(qtable)

# create hyperparameters
total_episodes = 50000  # Total episodes
total_test_episodes = 100  # Total test episodes
max_steps = 99  # Max steps per episode

learning_rate = 0.7  # Learning rate
gamma = 0.618  # Discounting rate

# Exploration parameters
epsilon = 1.0  # Exploration rate
max_epsilon = 1.0  # Exploration probability at start
min_epsilon = 0.01  # Minimum exploration probability
decay_rate = 0.01  # Exponential decay rate for exploration prob

# the specific algorithm

# todo: change Q learning to other algorithms
q_learning.run(env, qtable, epsilon, gamma, learning_rate, total_episodes, max_steps)

# use Q-table to play Taxi

env.reset()
rewards = []

for episode in range(total_test_episodes):
  state = env.reset()
  done = False
  total_rewards = 0
  # print("****************************************************")
  # print("EPISODE ", episode)
  
  for step in range(max_steps):
    # UNCOMMENT IT IF YOU WANT TO SEE OUR AGENT PLAYING
    # env.render()
    # Take the action (index) that have the maximum expected future reward given that state
    action = np.argmax(qtable[state, :])
    
    new_state, reward, done, info = env.step(action)
    
    total_rewards += reward
    
    if done:
      rewards.append(total_rewards)
      # print ("Score", total_rewards)
      break
    state = new_state
env.close()
print("Score over time: " + str(sum(rewards) / total_test_episodes))
