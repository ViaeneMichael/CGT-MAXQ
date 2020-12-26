import numpy as np
import gym

alpha = 0
gamma = 0
env = 0

# todo: primitive action values
v = []
# todo: non primitive action values
c = []
# todo: termination states
t = []

# state
class State:
  def __init__(self, episodes, nr_of_nodes, max_actions):
    self.__V = np.zeros((episodes, nr_of_nodes))
    # the nr_of_actions corresponds to the actions that have a value from a node below this node (this number is actually variable)
    # solution: use max_actions for now (which is 4)
    self.__C = np.zeros((episodes, nr_of_nodes, max_actions))
    self.__T = np.zeros(nr_of_nodes)
  
  def get_V(self, t, i):
    return self.__V[t][i.get_index()]
  
  def set_V(self, t, i, new_v):
    self.__V[t][i.get_index()] = new_v
  
  def get_C(self, t, i, j):
    return self.__C[t][i.get_index()][j.get_index()]
  
  def set_C(self, t, i, j, new_c):
    self.__C[t][i.get_index()][j.get_index()] = new_c
  
  def get_T(self, i):
    return self.__T[i.get_index()]
  
  def set_T(self, i, T):
    self.__T[i.get_index()] = T

# max node or Q node
class Node:
  def __init__(self, i, a):
    self.__idx = i
    # primitive if len(actions) == 1
    if type(a) == list:
      self.__actions = a
      self.__primitive = True
    else:
      self.__actions = [a]
      self.__primitive = False
  
  def get_actions(self):
    return self.__actions
  
  def set_actions(self, a):
    if type(a) == list:
      self.__actions = a
      self.__primitive = True
    else:
      self.__actions = [a]
      self.__primitive = False
  
  def get_index(self):
    return self.__idx
  
  def set_index(self, index):
    self.__idx = index
  
  def is_primitive(self):
    return self.__primitive

# def run(self, i, s):  # i is action number
#   if self.done:
#     i = 11  # to end recursion
#   self.done = False
#   if self.is_primitive(i):
#     self.new_s, r, self.done, _ = copy.copy(self.env.step(i))
#     self.r_sum += r
#     self.num_of_ac += 1
#     self.V[i, s] += self.alpha * (r - self.V[i, s])
#     return 1
#   elif i <= self.root:
#     count = 0
#     while not self.is_terminal(i, self.done):  # a is new action num
#       a = self.greed_act(i, s)
#       N = self.MAXQ_0(a, s)
#       self.V_copy = self.V.copy()
#       evaluate_res = self.evaluate(i, self.new_s)
#       self.C[i, s, a] += self.alpha * (self.gamma ** N * evaluate_res - self.C[i, s, a])
#       count += N
#       s = self.new_s
#     return count

# def run(env, qtable, epsilon, gamma, learning_rate, total_episodes, max_steps):
#   for episode in range(total_episodes):
#
#       # Reset the environment
#       s = env.reset()
#       step = 0
#       done = False
#
#       for step in range(max_steps):
#         # todo: 3. Choose an action a in the current world s (s)
#         print("select action")
#         action = ()
#
#         # todo: Take the action (a) and observe the outcome s(s') and reward (r)
#         new_state, reward, done, info = env.step(action)
#
#         # todo: update Q
#         print("update Q")
#
#         # todo: setup new s
#         s = new_state
#
#         # If done : finish episode
#         if done == True:
#            break

def is_primitive(max_node):
  return len(max_node.actions) == 1

# Greedy Execution of the MAXQ Graph.
def eval_max_node(t, i, s):
  if i.is_primitive():
    return s.get_V(t, i), i
  else:
    results = []
    actions = []
    # non primitive - every action of subtask
    for j in i.get_actions():
      v, j = eval_max_node(t, j, s)
      results.append(v + s.get_C(t, i, j))
      actions.append(j)  # to know which action responds to which result
    
    index = np.argmax(results)
    best_action = actions[index]
    
    return s.get_V(t, best_action), best_action

# i: max node
# s: s
def maxQ_0(t, i, s):
  if i.is_primitive():
    # observe result s' (I think nothing needs to change here -- done?)
    reward, action = eval_max_node(t, i, s)
    
    # todo: alpha gradually decreases to zero in the limit
    new_v = (1 - alpha) * s.get_V(t, i) + alpha * reward
    
    s.set_V(t + 1, i, s, new_v)
    return 1
  else:
    count = 0
    while not s.get_T(i):
      # choose action a according to the current exploration policy (hierarchical policy)
      v_t, a = eval_max_node(t, i, s)
      
      N = maxQ_0(t, a, s)
      
      # observe result state s' (What do I do here?) -- I think we don't do anything here (because I implmented state is object)
      # new_state = s
      
      new_c = (1 - alpha) * s.get_C(t, i, a) + alpha * np.pow(gamma, N) * v_t
      
      # push new C (t + 1)
      s.set_C(t + 1, i, a, new_c)
      
      count += N
      
      # in ons geval word de nieuwe s opgeslagen in states (ga naar volgende s?)
      # s = new_state
    return count

# Main

# todo: change epsiode shizzle later: the epsiodes variable is not always the same, depends on how fast it learns...
# I can change this in state:
# current V, C, etc.
# next V, C, etc.

def run(env, episodes):
  action_size = env.action_space.n
  state_size = env.observation_space.n
  
  # todo: write function that checks max_actions
  max_actions = 4
  
  # start with root node (0) and starting state s_0 (0)
  state = State(episodes, action_size, max_actions)  # starting state
  Root = 0  # starting action
  
  node = Node(0, Root)
  
  maxQ_0(0, node, state)

def test():
  env = gym.make('Taxi-v3')
  env.reset()
  for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
  env.close()
  
  action_space = env.action_space
  print(action_space)
  
  state_space = env.observation_space
  print(state_space)
  
  action_size = action_space.n
  state_size = env.observation_space.n
  
  print(action_size)
  print(state_size)


test()
