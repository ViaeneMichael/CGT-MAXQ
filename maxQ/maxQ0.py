from maxQ.agent import Agent
import numpy as np

# e-Greedy Execution of the MAXQ Graph.
def epsilon_greedy(maxnode, agent, s, args):
  epsilon = args[0]
  Q = []
  actions = []
  
  for j in maxnode.child_nodes:
    j = agent.graph[j]
    if j.primitive or not j.terminal(s):
      val = j.get_V(s) + maxnode.get_C(s, j)
      Q.append(val)
      actions.append(j)
  
  best_action_idx = np.argmax(Q)
  
  if np.random.rand(1) < epsilon:
    maxnode = np.random.choice(actions)
    return maxnode
  else:
    return actions[best_action_idx]

# evaluation of node
# todo: is this correct
def eval(maxnode, agent, old_state, new_state):
  if maxnode.primitive:
    return maxnode.get_V(old_state)
  else:
    for action1 in maxnode.child_nodes:
      print(maxnode.child_nodes)
      action = agent.graph[action1]
      # maxnode.get_V(old_state)
      action.set_V(new_state, action.get_V(old_state))  # Copy V from old value
      action.set_V(new_state, eval(maxnode, agent, old_state, new_state))
    
    Q = np.arange(0)
    nodes = np.arange(0)
    for action2 in maxnode.child_nodes:
      action = agent.graph[action2]
      Q = np.concatenate((Q, [action.get_V(new_state)]))
      nodes = np.concatenate(action)
    
    max_arg = np.argmax(Q)
    return nodes[max_arg].get_V(new_state)

# todo: is this correct
def maxQ0(agent, maxnode, state):
  count = 0
  if maxnode.primitive:
    agent.new_state, reward, done, info = agent.env.step(maxnode.action_index)
    temp_v = (1 - agent.alpha) * maxnode.get_V(state) + agent.alpha * reward
    maxnode.set_V(state, temp_v)
    return 1
  else:
    while not maxnode.terminal(state):
      action = maxnode.pick_action(epsilon_greedy, agent, state, [0.01])
      N = maxQ0(agent, action, state)
      v_t = eval(maxnode, agent, state, agent.new_state)
      new_c = (1 - agent.alpha) * maxnode.get_C(state, action) + agent.alpha * agent.gamma ** N * v_t
      maxnode.set_C(state, action, new_c)
      count += N
      state = agent.new_state
    return count

def run_game(env, episodes, alpha, gamma):
  agent = Agent(env, alpha, gamma, env.decode)
  rewards = []
  for j in range(episodes):
    
    # reset
    env.reset()
    agent.reward_sum = 0
    agent.new_state = env.s
    
    maxQ0(agent, agent.root, env.s)
    rewards.append(agent.reward_sum)
    
    if (j % 1000 == 0):
      print(j)
  
  np.save("saves\Qmax_{}".format(episodes), rewards)
  return rewards

# Make State class that can be indexed -> use as keys for C_vals in MaxNode
# State in OpenAI Taxi is four-tuple(taxi_row,taxi_col,pass_loc,dest_idx)
# Can be computed to single index: 
### def encode(maxnode, taxi_row, taxi_col, pass_loc, dest_idx):
###     # (5) 5, 5, 4 
###     maxnode = taxi_row
###     maxnode *= 5
###     maxnode += taxi_col
###     maxnode *= 5
###     maxnode += pass_loc
###     maxnode *= 4
###     maxnode += dest_idx
###     return maxnode
### def decode(maxnode, maxnode):
###     out = [] 
###     out.append(maxnode % 4)
###     maxnode = maxnode // 4
###     out.append(maxnode % 5)
###     maxnode = maxnode // 5
###     out.append(maxnode % 5)
###     maxnode = maxnode // 5
###     out.append(maxnode)
###     assert 0 <= maxnode < 5
###     return reversed(out)