import numpy as np

class Demonstration():
    def __init__(self):
        self.demonstrations = None # C demonstrations, D features
        self.rationality = None 
        pass

    def get_probability_of_input(self, omegas):
        '''
        args:
            omegas - N samples, D dimensions
        '''
        feature_sums = np.sum(self.demonstrations, axis=0) #1 by D features
        rewards = np.dot(feature_sums, omegas.T) # 1 reward , N samples
        return np.exp(rewards * self.rationality) / np.sum(np.exp(rewards * self.rationality))

    def tell_input(self, demos, rationality=.1):
        self.demonstrations = demos
        self.rationality = rationality