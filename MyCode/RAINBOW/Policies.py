import numpy as np
import numpy.random as npr

# For use with most policies
def epsGreedyActionSlection(qValues, eps):
    if npr.rand() < eps:  # Pick random action w/prob eps
        return npr.randint(qValues.shape[0])
    # Otherwise, pick best action, break ties at random
    actionIndex = np.argwhere(qValues==qValues.max()).flatten()
    return npr.choice(actionIndex)




# Basic epsilon greedy policy
class EpsilonGreedyPolicy(object):
    def __init__(self, eps=0):
        self.eps = eps

    def selectAction(self, qValues):
        return epsGreedyActionSlection(qValues, self.eps)