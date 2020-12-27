import numpy as np

class MaxNode:
  def __init__(self, action_index):
    # dictionary with (states:{actions:vals}) as keys and C_val as value
    # if we go get maxnode C_value and the key combination is not yet in the dictionary - initialize it to 0 (is done in the get_Cvals)
    self.C_vals = {}
    # dictionary with states as keys and Q_val as value
    # todo: ask copy V ?
    self.V_vals = {}
    self.action_index = action_index
    self.childNodes = []
    self.primitive = True
    self.decoder = lambda state: state
  
  def get_V(self, state):
    if state in self.V_vals:
      return self.V_vals[state]
    else:
      self.V_vals[state] = 0
    return self.V_vals[state]
  
  def set_V(self, state, val):
    self.V_vals[state] = val
  
  def get_C(self, state, action):
    if state in self.C_vals and action in self.C_vals[state]:
      return self.C_vals[state][action]
    elif state in self.C_vals:
      self.C_vals[state][action] = 0
    else:
      self.C_vals[state] = {}
      self.C_vals[state][action] = 0
    return self.C_vals[state][action]
  
  def set_C(self, state, action, val):
    self.C_vals[state][action] = val
  
  def set_decoder(self, state_decoder):
    self.decoder = state_decoder
  
  def addChildNode(self, action):
    self.primitive = False
    self.childNodes.append(action)
    
  def terminal(self, state):
    RGBY = [(0, 0), (0, 4), (4, 0), (4, 3)]
    taxirow, taxicol, passidx, destidx = list(self.decoder(state))
    taxiloc = (taxirow, taxicol)
    
    if self.get_action() == 10:
      return False
    elif self.get_action() == 9:
      return passidx < 4
    elif self.get_action() == 8:
      return passidx >= 4
    elif self.get_action() == 7:
      return passidx >= 4 and taxiloc == RGBY[destidx]
    elif self.get_action() == 6:
      return passidx < 4 and taxiloc == RGBY[passidx]
    elif self.primitive:
      return True
    
  def pîck_action(self, action_selection_method, state, args):
    return action_selection_method(self, state, args)
  
  
