import gym
import numpy as np
from MaxQ0 import maxQ0
from MaxQQ import maxQQ
from Polling import polling
import matplotlib.pyplot as plt

# qtable = np.zeros((state_size, action_size))

# print(qtable)

# use Q-table to play Taxi
# def run_simulation():
#   env.reset()
#   rewards = []
#
#   for episode in range(total_test_episodes):
#     state = env.reset()
#     done = False
#     total_rewards = 0
#
#     for step in range(max_steps):
#       # render agent learning
#       # env.render()
#
#       # Take the maxnode (index) that has the maximum expected future reward given that s
#       action = np.argmax(qtable[state, :])
#
#       new_state, reward, done, info = env.step(action)
#
#       total_rewards += reward
#
#       if done:
#         rewards.append(total_rewards)
#         # print ("Score", total_rewards)
#         break
#       state = new_state
#   env.close()
#   print("Score over time: " + str(sum(rewards) / total_test_episodes))

def show_plot(algorithm, trails, episodes):
  rewards = np.load(".\saves\{}_{}_{}.npy".format(algorithm, trails, episodes))
  reward_sequence = np.sum(rewards, axis=0) / trails
  
  # learning plot
  plt.figure(figsize=(15, 7.5))
  plt.plot(reward_sequence)
  plt.xlabel('episodes')
  plt.ylabel('reward / step')
  
  plt.savefig("./plots/{}_{}_{}".format(algorithm, trails, episodes))
  
  plt.show()
  

# Main: specify algorithm

# Q-learning
# create hyperparameters
# total_episodes = 20000  # Total episodes
# total_test_episodes = 1000  # Total states episodes
# max_steps = 99  # Max steps per episode
#
# learning_rate = 0.7  # Learning rate
# gamma = 0.618  # Discounting rate
#
# # Exploration parameters
# epsilon = 1.0  # Exploration rate
# max_epsilon = 1.0  # Exploration probability at start
# min_epsilon = 0.01  # Minimum exploration probability
# decay_rate = 0.01  # Exponential decay rate for exploration prob

# q_learning.run(env, qtable, min_epsilon, epsilon, max_epsilon, gamma, learning_rate,decay_rate, total_episodes, max_steps)

# Max Q learning
env = gym.make("Taxi-v3")

trails = 1
maxq_episodes = 50000  # maxq0 and maxqq 50 000 episodes
polling_episodes = 5000
gamma = 0.5

# print("-----------------------MAXQ-0 started-------------------------------")
# r_maxQ0 = maxQ0.run_game(env, trails, maxq_episodes, gamma)
# print("-----------------------MAXQ-Q started-------------------------------")
# r_maxQQ = maxQQ.run_game(env, trails, maxq_episodes, gamma)
print("-----------------------Polling started------------------------------")
polling = polling.run_game(env, trails, polling_episodes, gamma)

# show_plot("maxq0", trails, maxq_episodes)
# show_plot("maxqq",trails, maxq_episodes)
show_plot("polling", trails, polling_episodes)
