import copy

def run(self, i, s):  # i is action number
  if self.done:
    i = 11  # to end recursion
  self.done = False
  if self.is_primitive(i):
    self.new_s, r, self.done, _ = copy.copy(self.env.step(i))
    self.r_sum += r
    self.num_of_ac += 1
    self.V[i, s] += self.alpha * (r - self.V[i, s])
    return 1
  elif i <= self.root:
    count = 0
    while not self.is_terminal(i, self.done):  # a is new action num
      a = self.greed_act(i, s)
      N = self.MAXQ_0(a, s)
      self.V_copy = self.V.copy()
      evaluate_res = self.evaluate(i, self.new_s)
      self.C[i, s, a] += self.alpha * (self.gamma ** N * evaluate_res - self.C[i, s, a])
      count += N
      s = self.new_s
    return count
