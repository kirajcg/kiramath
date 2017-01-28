from . import rand
from . import stats
from . import linalg


class Markov(object):

    # Initialize object with transition matrix
    def __init__(self, transmat):
        self.transmat = transmat

    # Goes nowhere, does nothing
    def state(self, nr):
        P = self.transmat[nr]
        U = rand.random(mx=max(P))[0]

        for i in stats.shuffle(list(range(len(P)))):
            if U < P[i]:
                try:
                    print("Moving to state %i" % i)
                    return self.state(i)
                except RecursionError:
                    return

    # Computes equilibrium distribution as eigenvector of transition matrix
    def equil(self):
        E = linalg.eig(self.transmat)[1]
        return [e/sum(E) for e in E]
