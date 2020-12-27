class MaxNode:
    def __init__(self, action_index):
        # dictionary with (states:{actions:vals}) as keys and C_val as value
        # if we go get a C_value and the key combination is not yet in the dictionary - initialize it to 0 (is done in the get_Cvals)
        self.C_vals={}
        #dictionary with states as keys and Q_val as value
        self.V_vals={}
        self.action_index=action_index
        self.childNodes=[]
        self.primitive=True
    
    def getAction(self):
        return self.action_index

    def get_V(self,state):
        if state in V_vals:
            return self.V_vals[state]
        else:
            self.V_vals[state]=0
        return self.V_vals[state]

    def set_V(self,state,val):
        self.V_vals[state]=val

    def get_C(self,state,action):
        if state in C_vals and action in C_vals[state]:
           return self.C_vals[state][action]
        elif state in C_vals:
            self.C_vals[state][action] = 0
        else:
            self.C_vals[state]={}
            self.C_vals[state][action]=0
        return self.C_vals[state][action]
            

    def set_C(self,state,action,val):
        self.C_vals[state][action]=val

    def addChildNode(self,action):
        self.primitive=False
        self.childNodes.append(action)

class Agent:
    def __init__(self,alpha,gamma):
        maxRoot = MaxNode(10)
        maxGet=MaxNode(8) 
        maxPut=MaxNode(9)
        pickup=MaxNode(4)
        putdown=MaxNode(5)
        maxNavGet=MaxNode(6)
        maxNavPut=MaxNode(7)
        north=MaxNode(1)
        east=MaxNode(2)
        south=MaxNode(0)
        west=MaxNode(3)
        maxRoot.addChildNode(maxGet.getAction)
        maxRoot.addChildNode(maxPut.getAction)
        maxGet.addChildNode(pickup.getAction)
        maxGet.addChildNode(maxNavGet.getAction)
        maxPut.addChildNode(maxNavPut.getAction)
        maxPut.addChildNode(putdown.getAction)
        maxNavGet.addChildNode(north.getAction)
        maxNavGet.addChildNode(east.getAction)
        maxNavGet.addChildNode(south.getAction)
        maxNavGet.addChildNode(west.getAction)

        maxNavPut.addChildNode(north.getAction)
        maxNavPut.addChildNode(east.getAction)
        maxNavPut.addChildNode(south.getAction)
        maxNavPut.addChildNode(west.getAction)

        self.graph=maxRoot
        self.reward_sum=0
        self.alpha=alpha
        self.gamma = gamma

    def getGamma(self):
        return self.gamma

    def getAlpha(self):
        return self.alpha

def maxQ0(agent,maxnode,state):
    count = 0
    if maxnode.primitive:
        new_state, reward, done, info = agent.env.step(maxnode.action_index)
        temp_v = (1-agent.getAlpha())*maxnode.get_V(state)+agent.getAlpha()*reward
        return 1
    else:
        #TODO: implement terminal state in maxnode class
        while not maxnode.terminal(state):
            #TODO: implement pickaction in maxnode class (idea is to be able to give epsilon-greedy, blotmann or whatever as argument and the action is chosen using that function)
            action = maxnode.pickAction(methodofpicking)
            N = maxQ0(agent,action,state)
            new_state,reward,done,info = agent.env.step(maxnode.action_index)
            #TODO: update C-values
            count += N
            state = new_state
        return count

def run_game():
    env = initializeEnvironment()
    agent=Agent()
    maxQ0(agent, agent.graph,env.s)
    
        
        


# Make State class that can be indexed -> use as keys for C_vals in MaxNode
# State in OpenAI Taxi is four-tuple(taxi_row,taxi_col,pass_loc,dest_idx)
# Can be computed to single index: 
### def encode(self, taxi_row, taxi_col, pass_loc, dest_idx): 
###     # (5) 5, 5, 4 
###     i = taxi_row 
###     i *= 5 
###     i += taxi_col 
###     i *= 5 
###     i += pass_loc 
###     i *= 4 
###     i += dest_idx 
###     return i 
### def decode(self, i): 
###     out = [] 
###     out.append(i % 4) 
###     i = i // 4 
###     out.append(i % 5) 
###     i = i // 5 
###     out.append(i % 5) 
###     i = i // 5 
###     out.append(i) 
###     assert 0 <= i < 5 
###     return reversed(out) 
