# failure probability exp(-kappa)

from math import log, sqrt, ceil
import sympy

def twolevel(N, M, B, is_tight, kappa = 27.8):
    x = sympy.Symbol('x')
    g = x**2*(1-x)/(1+x)**2/(1+x/2)/(1+2*x)
    y = g**2-4*B*B*(1+2*x)*N/M**3*(kappa+2+1.5*log(N/M))**2
    res = sympy.solvers.solveset(y, x, sympy.Interval(0,1))
    if len(res)==0:
        raise ValueError("N too large!")
    beta = min(res)
    alpha = (kappa+2+log(N))*4*(1+beta)*(1+2*beta)/beta/beta/M 
    p = ceil(sqrt((1+2*beta)*N/M))
    _alpha, _beta, _p, _cost = onelevel(alpha*N, M, B, False, kappa+1)
    print("alpha'=%f, beta'=%f, p'=%d" % (_alpha, _beta, _p))
    if is_tight:
        cost = 9+7*alpha+8*beta
    else:
        cost = 8+(6+B)*alpha+10*beta
    return alpha, beta, p, cost


def onelevel(N, M, B, is_tight, kappa = 27.8):
    x = sympy.Symbol('x')
    g = x**2*(1-x)/(1+x)**2/(1+x/2)/(1+2*x)
    y = g-2*(1+2*x)*N*B/M/M*(kappa+1+2*log(N/M))
    res = sympy.solvers.solveset(y, x, sympy.Interval(0,1))
    if len(res)==0:
        raise ValueError("N too large!")
    beta = min(res)
    alpha = (kappa+1+log(N))*4*(1+beta)*(1+2*beta)/beta/beta/M
    if alpha * N > M - B:
        raise ValueError("N too large!")
    p = ceil((1+2*beta)*N/M)
    if is_tight:
        cost = 7+4*beta
    else:
        cost = 6+6*beta+alpha*B

    return alpha, beta, p, cost
        
