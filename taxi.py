import gym
import numpy as np
import q_learning
import maxQ0_old

# create environement
env = gym.make("Taxi-v3")
env.render()

# create Q-table and initialize it
action_size = env.action_space.n
# print("Action size ", action_size)

state_size = env.observation_space.n
# print("State size ", state_size)

qtable = np.zeros((state_size, action_size))
# print(qtable)

# create hyperparameters
total_episodes = 5001  # Total episodes
total_test_episodes = 10  # Total states episodes
max_steps = 99  # Max steps per episode

learning_rate = 0.7  # Learning rate
gamma = 0.618  # Discounting rate

# Exploration parameters
epsilon = 1.0  # Exploration rate
max_epsilon = 1.0  # Exploration probability at start
min_epsilon = 0.01  # Minimum exploration probability
decay_rate = 0.01  # Exponential decay rate for exploration prob

# the specific algorithm

# Q-learning
# q_learning.run(env, qtable, min_epsilon, epsilon, max_epsilon, gamma, learning_rate,decay_rate, total_episodes, max_steps)

# Max Q learning (without save data or fresh run)
#maxQ0.run(env, total_episodes)

# with save data
rewards = np.load("saves\Qmax_{}.npy".format(total_episodes))

maxQ0_old.show_plot(rewards)

# use Q-table to play Taxi
def run_simulation():
  env.reset()
  rewards = []
  
  for episode in range(total_test_episodes):
    state = env.reset()
    done = False
    total_rewards = 0
    
    for step in range(max_steps):
      # render agent learning
      # env.render()
      
      # Take the maxnode (index) that has the maximum expected future reward given that s
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
