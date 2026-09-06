import numpy as np
import scipy.optimize as opt
from scipy.spatial import KDTree

class InfoGainQueryGenerator():
    def __init__(self, bounds, discrete_set=None):
        '''
        
        '''
        self.lower_bounds = [bound[0] for bound in bounds] #lower bounds of query space
        self.upper_bounds = [bound[1] for bound in bounds] # upper bounds of query space
        self.bounds = bounds
        self.dimension = len(bounds) # number of dimensions per query
        self.space_of_queries = None
        
        if discrete_set is not None:
            self.space_of_queries = discrete_set #the set of things that can be chosen
            self.tree = KDTree(discrete_set)

    def _info_gain(self, reward_parameterization, input_model, query):
        '''

        '''
        return reward_parameterization.get_best_entropy(query, input_model) - \
                reward_parameterization.get_human_entropy(query, input_model)

    def get_query(self, number_queries, reward_parameterization, input_model):
        '''
        
        '''
        if self.space_of_queries is None:
            #continuous case
            return self._get_continuous_query(number_queries, reward_parameterization, input_model)
        else:
            #discrete case
            return self._get_discrete_query(number_queries, reward_parameterization, input_model)

    def _get_continuous_query(self, number_queries, reward_parameterization, input_model):
        '''
        
        '''
        x0 = np.random.uniform(low= self.lower_bounds * number_queries, 
                                high= self.upper_bounds * number_queries,
                                size=(self.dimension * number_queries))

        def objective(x):
            x = x.reshape(number_queries, self.dimension)
            return -self._info_gain(reward_parameterization, input_model, x)

        opt_res = opt.fmin_l_bfgs_b(objective, 
                                    x0=x0, 
                                    bounds=self.bounds * number_queries, 
                                    approx_grad=True, 
                                    iprint=-1)

        query = opt_res[0].reshape((number_queries, self.dimension))
        return query
    
    def _get_discrete_query(self, number_queries, reward_parameterization, input_model):
        '''

        '''
        assert number_queries < len(self.space_of_queries)

        return None