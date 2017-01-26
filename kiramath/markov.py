from . import rand
from . import stats


class Markov(object):

    def __init__(self, transmat):
        self.transmat = transmat

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
