# failure probability exp(-kappa)
# usually we take kappa=40 or 80
# all small parameters alpha, beta < 0.5

from math import log, log2, e, exp, sqrt, ceil

def merge_sort_cost(N, M, B):
    return 2*N/B*ceil(log(N/B,M/B))


class oqsort:
    def __init__(self, M, B, kappa):
        self.M, self.B, self.kappa = M, B, kappa
        
    def get_pmax(self, N):
        M, B, kappa = self.M, self.B, self.kappa
        pmax = M/(48*(log(log(N))+kappa)*B)
        return pmax

    def get_beta(self, p, N):
        M, B, kappa = self.M, self.B, self.kappa
        C = 4*p*B/M*(log(p*N/M)+log(log(N))+kappa)
        beta = 2*C+sqrt(4*C*C+C)
        return beta

    def IO_cost(self, p, N):
        M, B, kappa = self.M, self.B, self.kappa
        alpha = 1/p/B
        beta = self.get_beta(p, N)
        cost = (1+2*beta)*2*N/B*ceil(log((1+2*beta)*N/B,p))
        cost += (1+2*alpha)*N/B+4*(1+2*beta)*N/M
        cost += 2*N/B
        cost /= 1-alpha
        return cost

    def search_p(self, N):
        M, B, kappa = self.M, self.B, self.kappa
        pmax = self.get_pmax(N)
        p_range = range(2, ceil(pmax))
        p = min(p_range, key=lambda p:self.IO_cost(p, N))
        return p

    def print_all(self, N):
        M, B, kappa = self.M, self.B, self.kappa
        p = self.search_p(N)
        alpha = 1/p/B
        beta = self.get_beta(p, N)
        cost = self.IO_cost(p, N)
        print("alpha=%.3f,beta=%.3f,p=%d,cost=%.3f"%(alpha,beta,p,cost))
        


        
