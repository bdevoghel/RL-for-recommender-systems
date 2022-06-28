from gym.spaces import Box

class Agent:
    def __init__(self, action_space: Box):
        self.action_space = action_space
    
    def select_action(self, state):
        return self.action_space.sample()

    def learn(self, memory):
        pass

class Memory:
    def push(*args):
        pass
