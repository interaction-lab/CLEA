import numpy as np

class WeakPreferenceChoice():
    def __init__(self, delta):
        #TODO: find default values for delta
        self.query = None # [N choice, D dimensions]
        self.choice = None # integer
        self.delta = delta

    def get_probability_of_input(self, omegas):

        assert self.query is not None
        assert len(self.query) == 2 # for now, this is for two choices

        rewards = np.dot(self.query , omegas.T)

        p1 = np.reciprocal(1 + np.exp(self.delta + np.matmul(np.array([-1,1]), rewards)))
        if self.choice == 0:
             return p1

        p2 = np.reciprocal(1 + np.exp(self.delta + np.matmul(np.array([1,-1]), rewards)))
        if self.choice == 1:
             return p2

        if self.choice not in range(len(self.query)):
            return np.exp(self.delta + 1)*(p1*p2)

        return None

    def tell_input(self, choice, query):
        self.choice = choice
        self.query = query

    def get_choice_probabilities(self, options, omegas):
        '''
        options - N options by D dimensions
        omegas - S omegas by D dimensions

        returns N options by S omegas (where each [:, s] is the probability
            distribution over the choices for the sth omega)
        '''
        rewards = np.dot(options, omegas.T) # N options by S omegas of the rewards for each option
        p1 = np.reciprocal(1 + np.exp(self.delta + np.matmul(np.array([-1,1]), rewards)))
        p2 = np.reciprocal(1 + np.exp(self.delta + np.matmul(np.array([1,-1]), rewards)))
        other = np.exp(self.delta + 1)*(p1*p2)
        # print(p1.shape, p2.shape, other.shape)
        # print(p1)
        return np.vstack([p1,p2,other])