import numpy as np

class Agent:
  def __init__(self, nr_of_nodes, nr_of_states, alpha, gamma, env):
    self.env = env

    self.V = np.zeros((nr_of_nodes, nr_of_states))
    self.C = np.zeros((nr_of_nodes, nr_of_states, nr_of_nodes))
    self.V_copy = self.V.copy()
    
    # consistend of max nodes and Q nodes
    s = self.south = 0
    n = self.north = 1
    e = self.east = 2
    w = self.west = 3
    pickup = self.pickup = 4
    dropoff = self.dropoff = 5
    gotoS = self.gotoS = 6
    gotoD = self.gotoD = 7
    get = self.get = 8
    put = self.put = 9
    root = self.root = 10
    
    self.alpha = alpha
    self.step = 0.0
    self.episode = 0.0
    self.gamma = gamma
    self.done = False
    self.__reward_sum = 0
    self.new_s = self.env.s
    
    # set()  # new empty object
    self.graph = [
      set(),  # south
      set(),  # north
      set(),  # east
      set(),  # west
      set(),  # pickup
      set(),  # dropoff
      {s, n, e, w},  # gotoSource
      {s, n, e, w},  # gotoDestination
      {pickup, gotoS},  # get -> pickup, gotoSource
      {dropoff, gotoD},  # put -> dropoff, gotoDestination
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
    elif a == self.gotoD:
      return passidx >= 4 and taxiloc == RGBY[destidx]
    elif a == self.gotoS:
      return passidx < 4 and taxiloc == RGBY[passidx]
    elif self.is_primitive(a):
      return True
  
  def reset(self):
    self.env.reset()
    self.set_reward_sum(0)
    self.done = False
    self.new_s = self.env.s
    self.step = 0

# e-Greedy Execution of the MAXQ Graph.
def epsilon_greedy(agent, i, s):
  e = 1 / np.sqrt(agent.episode)
  Q = []
  actions = []
  
  for j in agent.graph[i]:
    if agent.is_primitive(j) or not agent.is_terminal(j):
      val = agent.get_V(j, s) + agent.get_C(i, s, j)
      Q.append(val)
      actions.append(j)
  
  best_action_idx = np.argmax(Q)
  
  if np.random.rand(1) < e:
    action = np.random.choice(actions)
    # print("return: {} from {}".format(maxnode, actions))
    return action
  else:
    # print("return: {} from {}".format(best_action_idx, actions))
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
def maxQ_0(agent, i, s):
  if agent.done:
    i = 11  # end recursion
  agent.done = False
  if agent.is_primitive(i):
    # observe result s' (I think nothing needs to change here -- done?)
    
    # take maxnode maxnode
    agent.new_s, reward, agent.done, info = agent.env.step(i)
    agent.step += 1
    
    agent.set_reward_sum(agent.get_reward_sum() + reward)
    
    new_v = (1 - agent.alpha) * agent.get_V(i, s) + agent.alpha * reward
    agent.set_V(i, s, new_v)
    return 1
  elif i <= agent.root:
    count = 0
    while not agent.is_terminal(i):
      # choose maxnode maxnode according to the current exploration policy (hierarchical policy)
      a = epsilon_greedy(agent, i, s)
      N = maxQ_0(agent, a, s)
      agent.V_copy = agent.V.copy()
      v_t = eval(agent, i, agent.new_s)
      new_c = (1 - agent.alpha) * agent.get_C(i, s, a) + agent.alpha * agent.gamma ** N * v_t
      agent.set_C(i, s, a, new_c)
      count += N
      s = agent.new_s
    return count

# Main
def run_game(env, trails, episodes, alpha, gamma):
  # gotoSource + gotoDestination + put + get + root (number of non primitive actions)
  np_actions = 5
  nr_of_nodes = env.action_space.n + np_actions
  nr_of_states = env.observation_space.n
  
  taxi_agent = Agent(nr_of_nodes, nr_of_states, alpha, gamma, env)  # starting state
  
  result = np.zeros((trails, int(episodes / 10), 2))
  avgReward = np.zeros((10, 2))
  avgStep = np.zeros((10, 2))
  for i in range(trails):
    print("trail: {}".format(i))
    count = 0
    taxi_agent.episode = 1
    for j in range(episodes):
      
      # reset
      taxi_agent.reset()
      
      # algorithm
      maxQ_0(taxi_agent, taxi_agent.root, env.s)  # start with root node (0) and starting state s_0 (0)
      
      # add average reward
      avgReward[count] = taxi_agent.get_reward_sum()
      avgStep[count] = taxi_agent.step
      
      count += 1
      
      # average of 10 rewards and add to result
      if count >= 10:
        result[i][int(j / 10)] = (np.average(avgReward), np.average(avgStep))
        avgReward = np.zeros(10)
        avgStep = np.zeros(10)
        count = 0
      
      # print status
      if j % 1000 == 0:
        print("episode: {}".format(j))
        
      taxi_agent.episode += 1
  
  np.save(".\saves\maxq0_{}_{}".format(trails, episodes), result)
  return result
