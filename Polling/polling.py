import numpy as np
import copy
from MaxQ0 import maxQ0

class Action:
  def __init__(self, state):
    self.path = []
    self.state = state
    self.new_state = 0
    self.reward = 0
  
  def set_parents(self, parents):
    self.path = parents

class Agent:
  def __init__(self, nr_of_nodes, nr_of_states, alpha, gamma, env):
    self.env = env
    
    self.V = np.zeros((nr_of_nodes, nr_of_states))
    self.C = np.zeros((nr_of_nodes, nr_of_states, nr_of_nodes))
    self.V_copy = self.V.copy()
    
    self.distribution = np.zeros(10)
    
    # consistend of max nodes and Q nodes
    s = self.south = 0
    n = self.north = 1
    e = self.east = 2
    w = self.west = 3
    pickup = self.pickup = 4
    dropoff = self.dropoff = 5
    navigate = self.navigate = 6
    get = self.get = 7
    put = self.put = 8
    root = self.root = 9
    
    self.put_illegal = 0
    
    self.episode = 0.0
    self.step = 0.0
    self.alpha = alpha
    self.gamma = gamma
    self.done = False
    self.__reward_sum = 0
    self.new_s = copy.copy(self.env.s)
    self.from_get = False
    
    # set()  # new empty object
    self.graph = [
      set(),  # south
      set(),  # north
      set(),  # east
      set(),  # west
      set(),  # pickup
      set(),  # dropoff
      {s, n, e, w},  # navigate
      {pickup, navigate},  # get -> pickup, navigate
      {dropoff, navigate},  # put -> dropoff, gotoDestination
      {put, get},  # root -> put, get
    ]
  
  def get_V(self, i, s):
    return self.V[i, s]
  
  def set_V(self, i, s, new_v):
    self.V[i, s] = new_v
  
  def get_C(self, i, s, j):
    return self.C[i, s, j]
  
  def set_C(self, i, s, j, new_c):
    self.C[i, s, j] = new_c
  
  def get_reward_sum(self):
    return self.__reward_sum
  
  def set_reward_sum(self, reward_sum):
    self.__reward_sum = reward_sum
  
  def is_primitive(self, a):
    return a <= 5
  
  def is_terminal(self, a):
    RGBY = [(0, 0), (0, 4), (4, 0), (4, 3)]
    taxirow, taxicol, passidx, destidx = list(self.env.decode(self.env.s))
    taxiloc = (taxirow, taxicol)
    
    if self.done:
      return True
    elif a == self.root:
      return self.done
    elif a == self.put:
      return passidx < 4
    elif a == self.get:
      return passidx >= 4
    elif a == self.navigate:
      if self.from_get:
        return passidx < 4 and taxiloc == RGBY[passidx]
      else:
        return passidx >= 4 and taxiloc == RGBY[destidx]
    elif self.is_primitive(a):
      return True
  
  def polling_terminal(self, a):
    RGBY = [(0, 0), (0, 4), (4, 0), (4, 3)]
    taxirow, taxicol, passidx, destidx = list(self.env.decode(self.env.s))
    taxiloc = (taxirow, taxicol)
    
    if a == self.put:
      return passidx < 4
    elif a == self.get:
      return passidx >= 4
    elif a == self.navigate:
      if self.from_get:
        return passidx < 4 and taxiloc == RGBY[passidx]
      else:
        return passidx >= 4 and taxiloc == RGBY[destidx]
  
  def print_action(self, action):
    if action == 0:
      print("south - done: {}".format(self.done))
    elif action == 1:
      print("north - done: {}".format(self.done))
    elif action == 2:
      print("east - done: {}".format(self.done))
    elif action == 3:
      print("west - done: {}".format(self.done))
    elif action == 4:
      print("pickup - done: {}".format(self.done))
    elif action == 5:
      print("dropoff - done: {}".format(self.done))
    elif action == 6:
      print("gotoSource - done: {}".format(self.done))
    elif action == 7:
      print("get -> pickup, gotoSource - done: {}".format(self.done))
    elif action == 8:
      print("put -> dropoff, gotoDestination - done: {}".format(self.done))
    elif action == 9:
      print("root -> put, get")
  
  def update_V_C(self, actions):
    for i in range(len(actions) - 1):
      v_t = (1 - self.alpha) * self.get_V(i, actions[i].state) + self.alpha * (self.gamma ** i) * actions[i].reward
      self.set_V(actions[i].path[0], actions[i].state, v_t)
      for j in range(1, len(actions[i].path)):
        v_t = eval(self, actions[i].path[j], actions[i].new_state)
        new_c = (1 - self.alpha) * self.get_C(actions[i].path[j], actions[i].state,
                                              actions[i].path[j - 1]) + self.alpha * (self.gamma ** i) * v_t
        self.set_C(actions[i].path[j], actions[i].state, actions[i].path[j - 1], new_c)
  
  def reset_V_C(self, nr_of_nodes, nr_of_states):
    self.V = np.zeros((nr_of_nodes, nr_of_states))
    self.C = np.zeros((nr_of_nodes, nr_of_states, nr_of_nodes))
    self.V_copy = self.V.copy()
  
  def check_dropoff(self, reward):
    RGBY = [(0, 0), (0, 4), (4, 0), (4, 3)]
    taxirow, taxicol, passidx, destidx = list(self.env.decode(self.env.s))
    taxiloc = (taxirow, taxicol)
    if (taxiloc in RGBY) and passidx == 4:
      return -10
    else:
      return reward
  
  def reset(self):
    self.env.reset()
    self.set_reward_sum(0)
    self.done = False
    self.new_s = copy.copy(self.env.s)
    self.step = 0

# e-Greedy Execution of the MAXQ Graph.
def epsilon_greedy(agent, i, s):
  e = 1 / np.sqrt(agent.episode)
  Q = []
  actions = []
  
  for j in agent.graph[i]:
    # if agent.is_primitive(j) or not agent.polling_terminal(j):
      val = agent.get_V(j, s) + agent.get_C(i, s, j)
      Q.append(val)
      actions.append(j)
  
  best_action_idx = np.argmax(Q)
  
  if agent.episode % 1000 == 0:
    print("best action is {} in {} - {}".format(best_action_idx, Q, actions))
  
  if np.random.rand(1) < e:
    action = np.random.choice(actions)
    return action
  else:
    return actions[best_action_idx]

# evaluation of node
def eval(agent, a, s):
  if agent.is_primitive(a):
    return agent.V_copy[a, s]
  else:
    for j in agent.graph[a]:
      agent.V_copy[j, s] = eval(agent, j, s)
    Q = np.arange(0)
    for a2 in agent.graph[a]:
      Q = np.concatenate((Q, [agent.V_copy[a2, s]]))
    max_arg = np.argmax(Q)
    return agent.V_copy[max_arg, s]

# maxnode: max node
# s: state
def polling(agent, i, s, parents):
  agent.distribution[i] += 1
  # if agent.done:
  # i = 11  # end recursion
  # agent.done = False
  if agent.is_primitive(i):
    agent.new_s, reward, _, info = copy.copy(agent.env.step(i))
    
    reward = agent.check_dropoff(reward)
    
    # per 1000 episodes render taxi problem
    if agent.episode % 1000 == 0:
      agent.env.render()
      print("reward: {}".format(reward))
    
    if i == agent.pickup and not reward == -10:
      agent.picked_up = True
    
    if reward < -2:
      agent.put_illegal += 1
    # agent.done = True
    
    agent.step += 1
    agent.set_reward_sum(agent.get_reward_sum() + reward)
    
    new_v = (1 - agent.alpha) * agent.get_V(i, s) + agent.alpha * reward
    agent.set_V(i, s, new_v)
    
    # if agent.episode > 1500:
    #   agent.print_action(i)
    #   print("V={}".format(agent.get_V(i, s)))
    parents.append(i)
    return 1, parents, reward
  elif i <= agent.root:
    
    if i == agent.get:
      agent.from_get = True
    
    if agent.from_get and i == agent.put:
      agent.from_get = False
    
    count = 0
    a = epsilon_greedy(agent, i, s)
    N, n_parents, reward = polling(agent, a, s, parents)
    agent.V_copy = agent.V.copy()
    v_t = eval(agent, i, agent.new_s)
    # v_t = agent.get_V(a,agent.new_s)
    new_c = (1 - agent.alpha) * agent.get_C(i, s, a) + agent.alpha * (agent.gamma ** N) * v_t
    agent.set_C(i, s, a, new_c)
    count += N
    # s = agent.new_s
    n_parents.append(i)
    return count, n_parents, reward

# Main
def run_game(env, trails, episodes, alpha, gamma):
  # gotoSource + gotoDestination + put + get + root (number of non primitive actions)
  np_actions = 5
  nr_of_nodes = env.action_space.n + np_actions
  nr_of_states = env.observation_space.n
  
  taxi_agent = Agent(nr_of_nodes, nr_of_states, alpha, gamma, env)  # starting state
  
  moving_average = 10
  
  result = np.zeros((trails, int(episodes / moving_average), 2))
  avgReward = np.zeros(moving_average)
  avgStep = np.zeros(moving_average)
  for i in range(trails):
    print("trail: {}".format(i))
    count = 0
    taxi_agent.episode = 1
    taxi_agent.reset_V_C(nr_of_nodes, nr_of_states)
    for j in range(episodes):
      # reset
      taxi_agent.reset()
      
      # print first state
      if (j + 1) % 1000 == 0:
        print("===================== {} =====================".format(j + 1))
        env.render()
      
      # for solving convergence problem
      # if taxi_agent.episode < 1000:
      #   maxQ0.maxQ_0(taxi_agent, taxi_agent.root, env.s)
      # else:
      RGBY = [(0, 0), (0, 4), (4, 0), (4, 3)]
      taxirow, taxicol, passidx, destidx = list(taxi_agent.env.decode(taxi_agent.env.s))
      taxiloc = (taxirow, taxicol)
      
      actions = []
      
      # passidx can only be >= 4 when passenger is picked up because otherwise it would not be >= 4
      while not (passidx >= 4 and taxiloc == RGBY[destidx]) and taxi_agent.step < env._max_episode_steps:
        # algorthm
        action = Action(env.s)
        _, parents, r = polling(taxi_agent, taxi_agent.root, env.s, [])
        action.set_parents(parents)
        action.new_state = taxi_agent.new_s
        action.reward = r
        
        if len(actions) <= 5:
          actions.append(action)
        else:
          actions = actions[1:]
          actions.append(action)
        
        # print(actions[-1].path)
        # taxi_agent.update_V_C(actions)
        
        taxirow, taxicol, passidx, destidx = list(taxi_agent.env.decode(taxi_agent.env.s))
        taxiloc = (taxirow, taxicol)
      
      # add average reward
      avgReward[count] = taxi_agent.get_reward_sum()
      avgStep[count] = taxi_agent.step
      
      count += 1
      
      # average of 10 rewards and add to result
      if count >= moving_average:
        result[i][int(j / moving_average)] = (np.average(avgReward), np.average(avgStep))
        avgReward = np.zeros(moving_average)
        avgStep = np.zeros(moving_average)
        count = 0
      
      # print status
      if j % 10 == 0:
        print("episode: {}".format(j))
      
      # print("=========================================================================")
      
      # if j % 1500 == 0:
      #   print(taxi_agent.distribution / np.sum(taxi_agent.distribution))
      #   print(taxi_agent.put_illegal)
      #   taxi_agent.distribution = np.zeros(10)
      
      taxi_agent.episode += 1
    
    # if i % 5 == 0 and i != 0:
    #   np.save(".\saves\polling_{}_{}".format(i, episodes), result)
  
  # print("END RESULT")
  # print(taxi_agent.distribution / np.sum(taxi_agent.distribution))
  # print(taxi_agent.put_illegal)
  
  np.save(".\saves\polling_{}_{}".format(trails, episodes), result)
  return result
