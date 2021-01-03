import numpy as np
from collections import deque
import operator
import copy

class Agent:
  def __init__(self, nr_of_nodes, nr_of_states, alpha, gamma, env):
    self.env = env
    
    self.V = np.zeros((nr_of_nodes, nr_of_states))
    self.C = np.zeros((nr_of_nodes, nr_of_states, nr_of_nodes))
    self.C_tilde = np.zeros((nr_of_nodes, nr_of_states, nr_of_nodes))
    self.V_copy = self.V.copy()
    
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
    
    self.alpha = alpha
    self.step = 0.0
    self.episode = 0.0
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
      {dropoff, navigate},  # put -> dropoff, navigate
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
  
  def get_C_tilde(self, i, s, j):
    return self.C_tilde[i, s, j]
  
  def set_C_tilde(self, i, s, j, new_c_tilde):
    self.C_tilde[i, s, j] = new_c_tilde
  
  def get_reward_sum(self):
    return self.__reward_sum
  
  def set_reward_sum(self, reward_sum):
    self.__reward_sum = reward_sum
  
  def is_primitive(self, a):
    return a <= 5
  
  def get_taxi_location(self, s):
    taxirow, taxicol, passidx, destidx = list(self.env.decode(s))
    taxiloc = (taxirow, taxicol)
    return taxiloc
  
  def get_start_goal(self):
    RGBY = [(0, 0), (0, 4), (4, 0), (4, 3)]
    taxirow, taxicol, passidx, destidx = list(self.env.decode(self.env.s))
    print("passenger pos = {}".format(RGBY[passidx]))
    print("goal pos = {}".format(RGBY[destidx]))
  
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
  
  def reset_V_C(self, nr_of_nodes, nr_of_states):
    self.V = np.zeros((nr_of_nodes, nr_of_states))
    self.C = np.zeros((nr_of_nodes, nr_of_states, nr_of_nodes))
    self.C_tilde = np.zeros((nr_of_nodes, nr_of_states, nr_of_nodes))
    self.V_copy = self.V.copy()
  
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
    if agent.is_primitive(j) or not agent.is_terminal(j):
      val = agent.get_V(j, s) + agent.get_C(i, s, j)
      Q.append(val)
      actions.append(j)
  
  best_action_idx = np.argmax(Q)
  
  if np.random.rand(1) < e:
    action = np.random.choice(actions)
    return action
  else:
    return actions[best_action_idx]

def argmax_Q(agent, i, s):
  actions = []
  
  for j in agent.graph[i]:
    actions.append(j)
  
  qs = []
  for a in actions:
    V = agent.get_V(a, s)
    C = agent.get_C_tilde(i, s, a)
    qs.append(V + C)
  
  i, v = max(enumerate(qs), key=operator.itemgetter(1))
  return actions[i]

def R_tilde(agent, i):
  # if agent.is_terminal(i):
  #   return 1.0
  # else:
  return 0.0

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

def maxQ_Q(agent, i, s):
  seq = deque()
  if agent.done:
    i = 11  # end recursion
  agent.done = False
  if agent.is_primitive(i):
    agent.new_s, reward, agent.done, info = copy.copy(agent.env.step(i))
    
    agent.step += 1
    
    agent.set_reward_sum(agent.get_reward_sum() + reward)
    
    new_v = (1.0 - agent.alpha) * agent.get_V(i, s) + agent.alpha * reward
    agent.set_V(i, s, new_v)
    seq.appendleft(s)
    return seq
  elif i <= agent.root:
    
    if i == agent.get:
      agent.from_get = True
    
    if agent.from_get and i == agent.put:
      agent.from_get = False
    
    while not agent.is_terminal(i):
      
      a = epsilon_greedy(agent, i, s)
      childSeq = maxQ_Q(agent, a, s)
      
      a_opt = argmax_Q(agent, i, agent.new_s)
      
      N = 1
      
      for _s in childSeq:
        Q_tilde = agent.get_C_tilde(i, agent.new_s, a_opt) + agent.get_V(a_opt, _s)
        new_c_tilde = (1 - agent.alpha) * agent.get_C_tilde(i, _s, a) + agent.alpha * (agent.gamma ** N) * (
                R_tilde(agent, i) + Q_tilde)
        
        agent.set_C_tilde(i, _s, a, new_c_tilde)
        # update C value
        agent.V_copy = agent.V.copy()
        v_t = eval(agent, i, agent.new_s)
        new_c = (1 - agent.alpha) * agent.get_C(i, _s, a) + agent.alpha * (agent.gamma ** N) * (
                agent.get_C(i, agent.new_s, a_opt) + v_t)
        agent.set_C(i, _s, a, new_c)
        N = N + 1
      
      seq.extendleft(childSeq)
      s = agent.new_s
    
    return seq

# Main
def run_game(env, trails, episodes, alpha, gamma):
  # gotoSource + gotoDestination + put + get + root (number of non primitive actions)
  np_actions = 5
  nr_of_nodes = env.action_space.n + np_actions
  nr_of_states = env.observation_space.n
  
  taxi_agent = Agent(nr_of_nodes, nr_of_states, alpha, gamma, env)  # starting state
  
  result = np.zeros((trails, int(episodes / 10), 2))
  avgReward = np.zeros(10)
  avgStep = np.zeros(10)
  for i in range(trails):
    print("trail: {}".format(i))
    count = 0
    taxi_agent.episode = 1
    taxi_agent.reset_V_C(nr_of_nodes, nr_of_states)
    for j in range(episodes):
      
      # reset
      taxi_agent.reset()
      
      # algorithm
      maxQ_Q(taxi_agent, taxi_agent.root, env.s)  # start with root node (0) and starting state s_0 (0)
      
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
  
  np.save(".\saves\maxqq_{}_{}".format(trails, episodes), result)
  return result
