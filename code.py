# failure probability exp(-kappa)
# usually we take kappa=40 or 80
# all small parameters alpha, beta < 0.5

from math import log, log2, e, exp, sqrt, ceil
import sympy

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
        print("C: " + str(C))
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

class oqsort2:
	"""docstring for oqsort2"""
	def __init__(self, M, B, kappa):
		self.M, self.B, self.kappa = M, B, kappa

	def get_alpha(self, N):
		return self.M/N

	def get_beta(self, N):
		return sqrt((8*N*self.B/(M**2))*(self.kappa+2*log(N/self.M)))

	def get_p(self, N, beta):
		return ceil((1+2*beta)*N/self.M)

	def Nsmall(self, p, N, beta):
		expr1 = self.M/self.B 
		expr2 = ((1+2*beta)*N)/self.B 
		expr3 = (p * self.M)/self.B
		return (expr1 < expr2) and (expr2 < expr3)
		
class onelevel_oqsort:
    def __init__(self, N, M, B, kappa):
        self.N, self.M, self.B, self.kappa = N, M, B, kappa

    def get_alpha(self):
    	return self.M/self.N

    def get_p(self, beta):
        return ceil((1+2*beta)*self.N/self.M)

    def get_beta(self):
        N, M, B, kappa = self.N, self.M, self.B, self.kappa
        x = sympy.Symbol('x')
        y = x**2*(1-x)/(1+x)**2/(1+x/2)/(1+2*x)**2-4*N*B/M/M*(kappa+2*log(N/M))
        res = sympy.solveset(y, x, sympy.Interval(0,1))
        if len(res)==0:
            print("Inappropriate inputs!")
            return 0
        return min(res)

    # in terms of N/B. Optimal=4, Bucket sort=12
    def IO_cost(self):
        N, M, B, kappa = self.N, self.M, self.B, self.kappa
        beta = self.get_beta()
        return 5+5*M/N+6*beta+6/B


if __name__ == '__main__':
	# params = oqsort(400, 4, 0.2)
	# print(params.get_pmax(20))
	# params.print_all(20)
	# N = 9 *16*2**16
	# M, B, kappa = 16*2**16, 8, 28
	N = 9000000
	M, B, kappa = 1000000, 8, 28
	params = onelevel_oqsort(N, M, B, kappa)
	alpha = params.get_alpha()
	beta = params.get_beta()
	p = params.get_p(beta)

	print("alpha=%.3f,beta=%.3f,p=%d,N=%d,M=%d,B=%d"%(alpha,beta,p,N,M,B))

        


        
