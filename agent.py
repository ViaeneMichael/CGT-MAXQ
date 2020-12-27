from max_node import MaxNode

class Agent:
  def __init__(self, alpha, gamma, state_decoder):
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
    maxRoot.addChildNode(maxGet.get_action)
    maxRoot.addChildNode(maxPut.get_action)
    
    # get
    maxGet.addChildNode(pickup.get_action)
    maxGet.addChildNode(maxNavGet.get_action)
    
    # put
    maxPut.addChildNode(maxNavPut.get_action)
    maxPut.addChildNode(putdown.get_action)
    
    # gotoSource
    maxNavGet.addChildNode(north.get_action)
    maxNavGet.addChildNode(east.get_action)
    maxNavGet.addChildNode(south.get_action)
    maxNavGet.addChildNode(west.get_action)
    
    # gotoDestination
    maxNavPut.addChildNode(north.get_action)
    maxNavPut.addChildNode(east.get_action)
    maxNavPut.addChildNode(south.get_action)
    maxNavPut.addChildNode(west.get_action)
    
    self.graph = maxRoot
    self.reward_sum = 0
    self.alpha = alpha
    self.gamma = gamma
  
  def getGamma(self):
    return self.gamma
  
  def getAlpha(self):
    return self.alpha
