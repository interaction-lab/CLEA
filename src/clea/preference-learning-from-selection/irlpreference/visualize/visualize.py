import numpy as np
import matplotlib
import matplotlib.pyplot as plt

'''
draws a graph of the hypothesis space of linear weights for the features.
Limited to two features for visualization.
'''
def sketch_probabilities(w_samples, log_w_probabilities, true_w=None):
    w_probabilities = np.exp(log_w_probabilities) #convert to probabilities

    #make the color bar
    minima = np.min(w_probabilities)
    maxima = np.max(w_probabilities)
    norm = matplotlib.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
    mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.cm.Greys_r)
    colors=mapper.to_rgba(w_probabilities)

    #plot all the points
    plt.scatter(w_samples[:,0], w_samples[:,1], color=colors);
    
    #plot the direction of the actual preference (if this is known)
    if true_w is not None:
        plt.scatter(true_w[0], true_w[1])
        plt.scatter(0,0)

    #label the axes and give it a title
    plt.xlabel('w1');
    plt.ylabel('w2');
    plt.title('our hypothesis space of w');
    plt.gca().set_aspect(1.0) #sets ratio to 1 to 1
    plt.colorbar(mapper)
