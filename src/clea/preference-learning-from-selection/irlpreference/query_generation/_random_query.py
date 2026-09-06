import numpy as np

class RandomQueryGenerator():
    def __init__(self, bounds, discrete_set=None):
        '''
        
        '''
        self.lower_bounds = [bound[0] for bound in bounds] #lower bounds of query space
        self.upper_bounds = [bound[1] for bound in bounds] # upper bounds of query space
        self.dimension = len(bounds) # number of dimensions per query
        self.space_of_queries = discrete_set #the set of things that can be chosen

    def get_query(self, number_queries, reward_parameterization=None, input_model=None):
        '''
        
        '''
        if self.space_of_queries is None:
            #continuous case
            return self._get_continuous_query(number_queries)
        else:
            #discrete case
            return self._get_discrete_query(number_queries)

    def _get_continuous_query(self, number_queries):
        '''
        
        '''
        return np.random.uniform(low=self.lower_bounds,
                                high=self.upper_bounds,
                                size=(number_queries, self.dimension))
    
    def _get_discrete_query(self, number_queries):
        '''

        '''
        assert number_queries <= len(self.space_of_queries)

        indices = np.random.choice(np.arange(len(self.space_of_queries)),
                                size=number_queries,
                                replace=False)
        print(indices)                 
        return self.space_of_queries[indices]