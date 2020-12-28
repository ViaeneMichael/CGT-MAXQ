from legacy.agent import Agent
import numpy as np

# e-Greedy Execution of the MAXQ Graph.
def epsilon_greedy(maxnode, s, args):
  epsilon = args[0]
  Q = []
  actions = []
  
  for j in maxnode.child_nodes:
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
def eval(maxnode, old_state, new_state):
  if maxnode.primitive:
    return maxnode.get_V(old_state)
  else:
    for idx in np.arange(len(maxnode.child_nodes)):
      maxnode.child_nodes[idx].set_V(new_state, maxnode.child_nodes[idx].get_V(old_state))
      maxnode.child_nodes[idx].set_V(new_state, eval(maxnode.child_nodes[idx], old_state, new_state))

    Q = np.arange(0)
    nodes = []
    for action2 in maxnode.child_nodes:
      Q = np.concatenate((Q, [action2.get_V(new_state)]))
      nodes = np.append(nodes, action2)

    max_arg = np.argmax(Q)
    return nodes[max_arg].get_V(new_state)
  

# todo: is this correct
def maxQ0(agent, maxnode, state):
  if agent.done:
    print("tis gedaan!")
    return 0
  agent.done = False
  if maxnode.primitive:
    agent.new_state, reward, done, info = agent.env.step(maxnode.action_index)
    agent.done = done
    agent.reward_sum += reward
    temp_v = (1.0 - agent.alpha) * maxnode.get_V(state) + agent.alpha * reward
    maxnode.set_V(state, temp_v)
    return 1
  else:
    count = 0
    while not maxnode.terminal(state):
      print("In loop:")
      print(maxnode.terminal(state))
      action = maxnode.pick_action(epsilon_greedy, state, [0.01])
      print("action: {}".format(action.action_index))
      N = maxQ0(agent, action, state)
      maxnode.set_V(agent.new_state, maxnode.get_V(state)) # copy
      v_t = eval(maxnode, state, agent.new_state)
      new_c = (1 - agent.alpha) * maxnode.get_C(state, action) + agent.alpha * agent.gamma ** N * v_t
      maxnode.set_C(state, action, new_c)
      count += N
      state = agent.new_state
      print(count)
    return count

def run_game(env, episodes, alpha, gamma):
  agent = Agent(env, alpha, gamma, env.decode)
  rewards = []
  for j in range(episodes):
    
    # reset
    env.reset()
    agent.reward_sum = 0
    agent.new_state = env.s
    
    maxQ0(agent, agent.graph, env.s)
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
