import random
import numpy as np

def run(env, qtable, min_epsilon, epsilon, max_epsilon, gamma, learning_rate, decay_rate, total_episodes, max_steps):
  for episode in range(total_episodes):
    # Reset the environment
    state = env.reset()
    step = 0
    done = False
    
    for step in range(max_steps):
      # 3. Choose an action a in the current world s (s)
      ## First we randomize a number
      exp_exp_tradeoff = random.uniform(0, 1)
      
      ## If this number > greater than epsilon --> exploitation (taking the biggest Q value for this s)
      if exp_exp_tradeoff > epsilon:
        action = np.argmax(qtable[state, :])
      # Else doing a random choice --> exploration
      else:
        action = env.action_space.sample()
      
      # Take the action (a) and observe the outcome s(s') and reward (r)
      new_state, reward, done, info = env.step(action)
      
      # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
      qtable[state, action] = qtable[state, action] + \
                              learning_rate * \
                              (reward + gamma * np.max(qtable[new_state, :])
                               - qtable[state, action])
      
      # Our new s is s
      state = new_state
      
      # If done : finish episode
      if done == True:
        break
      
      # Reduce epsilon (because we need less and less exploration)
      epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
