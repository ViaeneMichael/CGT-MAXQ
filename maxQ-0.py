import copy
import numpy as np

alpha = 0
gamma = 0
env = 0

# primitive action values
v = []
# non primitive action values
c = []
# termination states
t = []

class State:
  def __init__(self, episodes, nr_of_nodes, max_actions):
    
    self.__V = np.zeros((episodes, nr_of_nodes))
    # the nr_of_actions corresponds to the actions that have a value from a node below this node (this number is actually variable)
    # solution: use max_actions for now (which is 4)
    self.__C = np.zeros((episodes, nr_of_nodes, max_actions))
    self.__T = np.zeros(nr_of_nodes)
  
  def get_V(self, t, i):
    return self.__V[t][i]
  
  def set_V(self, t, i, new_v):
    self.__V[t][i] = new_v
  
  def get_C(self, t, i, j):
    return self.__C[t][i][j]
  
  def set_C(self, t, i, j, new_c):
    self.__C[t][i][j] = new_c
  
  def get_T(self, i):
    return self.__T[i]
  
  def set_T(self, i, T):
    self.__T[i] = T

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

# checks if max_node is a primitive max_node


# def V(i,s):
#   return v[i][s]
#
# def C(i,s, j):
#   return c[i][s][j]
#
# def init_V(i,s):
#   print("init V")
#
# def init_C(i,s,j):
#   print("init C")
#
# def T(s):
#   return t[s]

def is_primitive(max_node):
  return True

# Greedy Execution of the MAXQ Graph.
def eval__max_node(i, s):
  return 0, 0

# i: max node
# s: s
def maxQ_0(t, i, s):
  if is_primitive(i):
    # todo: execute max_node i recieve r and observe result s'
    print("todo: get reward here!")
    reward, action = eval__max_node(i, s)
    new_state = 0
    
    # alpha gradually decreases to zero in the limit
    # reward got from last step
    new_v = (1 - alpha) * s.get_V(t, i, s) + alpha * reward
    
    # todo: push new V (t + 1)
    s.set_V(t + 1, i, s, new_v)
    return 1
  else:
    count = 0
    while not s.get_T(i):
      # choose action a according to the current exploration policy (hierarchical policy)
      a = 0  # todo: change this
      N = maxQ_0(s, a, s)
      
      # todo: observe result s s' (What do I do here?)
      new_state = s
      
      j = 0  # todo: find this chosen action
      
      
      v_t = 0 # todo: find this value
      
      new_c = (1 - alpha) * s.get_C(t, i, j) + alpha * np.pow(gamma, N) * v_t
      
      # todo: push new C (t + 1)
      s.set_C(t + 1, i, j, new_c)
      
      count += N
      
      # todo: in ons geval word de nieuwe s opgeslagen in states (ga naar volgende s?)
      s = new_state
    return count
  
  
# Main

# the epsiodes variable is not always the same, depends on how fast it learns...
# todo: change epsiode shizzle later
def run(env, episodes):
  action_size = env.action_space.n
  state_size = env.observation_space.n
  
  # todo: write function that checks max_actions
  max_actions = 4
  
  state = State(episodes, action_size, max_actions)
  
  # start with root node (0) and starting s s_0 (0)
  # recursive call, number of episodes unknown
  maxQ_0(0, 0, state)
