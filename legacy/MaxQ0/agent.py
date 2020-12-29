from legacy.MaxQ0.max_node import MaxNode

class Agent:
  def __init__(self, env, alpha, gamma, state_decoder):
    # non-primitive actions
    maxRoot = MaxNode(10)
    maxGet = MaxNode(8)
    maxPut = MaxNode(9)
    pickup = MaxNode(4)
    putdown = MaxNode(5)
    maxNavGet = MaxNode(6)
    maxNavPut = MaxNode(7)
    
    # primitive actions
    south = MaxNode(0)
    north = MaxNode(1)
    east = MaxNode(2)
    west = MaxNode(3)
    
    # todo: add decoder directly in constructor
    maxRoot.set_decoder(state_decoder)
    maxRoot.set_decoder(state_decoder)
    maxGet.set_decoder(state_decoder)
    maxPut.set_decoder(state_decoder)
    pickup.set_decoder(state_decoder)
    putdown.set_decoder(state_decoder)
    maxNavGet.set_decoder(state_decoder)
    maxNavPut.set_decoder(state_decoder)
    south.set_decoder(state_decoder)
    north.set_decoder(state_decoder)
    east.set_decoder(state_decoder)
    west.set_decoder(state_decoder)
    
    # root
    maxRoot.add_child_node(maxGet)
    maxRoot.add_child_node(maxPut)
    
    # get
    maxGet.add_child_node(pickup)
    maxGet.add_child_node(maxNavGet)
    
    # put
    maxPut.add_child_node(maxNavPut)
    maxPut.add_child_node(putdown)
    
    # gotoSource
    maxNavGet.add_child_node(north)
    maxNavGet.add_child_node(east)
    maxNavGet.add_child_node(south)
    maxNavGet.add_child_node(west)
    
    # gotoDestination
    maxNavPut.add_child_node(north)
    maxNavPut.add_child_node(east)
    maxNavPut.add_child_node(south)
    maxNavPut.add_child_node(west)
    
    self.env = env
    self.new_state = env.s
    self.done = False
    self.graph = maxRoot
    self.reward_sum = 0
    self.alpha = alpha
    self.gamma = gamma
  
  def getGamma(self):
    return self.gamma
  
  def getAlpha(self):
    return self.alpha
