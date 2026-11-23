import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.special import zeta
from numpy.linalg import inv
from numpy import array
from numpy import sqrt
from numpy import pi
from matplotlib import pyplot as plt

do = False
#do = True
# 0 -- minus   1 -- plus

d = 3
#a = np.array( [[1,0.2],[0.2,1]] )
a = np.array( [[0.1,0.5],[0.5,0.1]] )
a00, a01, a11 = a[0][0], a[0][1], a[1][1]
a10 = a01
v = 1/10

u0_max = -v**2
u1_max = 0
u0_min = -100
u1_min = -100

def extrapolate_grid(x, df):
    xs = df.index.values
    ys = df['y'].values
    dx = xs[1]-xs[0]
    i = int( (x-xs[0]) / dx )
    i = max(i, 0)
    i = min(i, len(xs)-2)
    return ys[i] + (x - xs[i]) * (ys[i+1] - ys[i])/dx

def export(f, x0, x1, filename):
    xs = np.linspace( x0, x1, 10000 )
    ys = [f(x) for x in xs]
    df = pd.DataFrame(ys, index=xs, columns=['y']) 
    df.to_csv(filename)

def _import(filename):
    df = pd.read_csv(filename, index_col=0)
    return lambda x: extrapolate_grid(x, df)

def calc(f, x0, x1, filename):
    if(do): export(f, x0, x1, filename)
    return _import(filename)

def invert(filename):
    df = pd.read_csv(filename, index_col=0)
    xs = df.index.values
    ys = df['y'].values
    if ys[-1] > ys[0]:
        res_xs = np.linspace(ys[0], ys[-1], 100000)
        res_ys = []
        i = 0
        for x in res_xs:
            while i < len(ys)-2 and ys[i+1] < x:
                i += 1
            res_ys.append( xs[i] + (x-ys[i]) * (xs[i+1]-xs[i]) / (ys[i+1]-ys[i]) )
            #res_ys.append( xs[i] )
    else:
        res_xs = np.linspace(ys[-1], ys[0], 100000)
        res_ys = []
        i = len(ys)-2
        for x in res_xs:
            while i > 0 and ys[i] > x:
                i -= 1
            res_ys.insert( xs[i] + (x-ys[i]) * (xs[i+1]-xs[i]) / (ys[i+1]-ys[i]), 0 )
            #res_ys.insert( xs[i] )
    df = pd.DataFrame(res_ys, index=res_xs, columns=['y'])
    df.to_csv('inverse_' + filename) 
    #plt.plot(res_xs, res_ys)
    #plt.show()
    
def _polylog(s, x):
    if x > 1:
        raise ValueError('Illegal polylog argument')
    if x == 1:
        if s == 3/2:
            return zeta(s)
        if s == 1/2:
            return np.inf
    if x == 0:
        return 0
    if s == 1:
        return -np.log(1-x)
    if s == 0:
        return x/(1-x)
    if s == -1:
        return x/(1-x)**2
    result = x
    n = 2
    while True:
        rest = x**n / (1-x) / n**s
        if rest/result < 0.0001:
            return result
        result += x**n / n**s
        n += 1

def g_integrand(r, s, u):
    return 2 * r * polylog(d/2-1, np.exp(-r**2 -2*v*s*r + u))

def g(s, u):
    integrand = lambda r: g_integrand(r, s, u)
    return quad( integrand, 0, np.inf, epsabs = 0, epsrel=0.001 )[0]

def logz_integrand(r, s, u):
    return 2 * r * polylog(d/2, np.exp(-r**2 -2*v*s*r + u))

def h(s, u):
    integrand = lambda r: logz_integrand(r, s, u)
    return quad( integrand, 0, np.inf, epsabs = 0, epsrel=0.001 )[0]

def logz0(u0):
    return h(-1,u0)

def logz1(u1):
    return h(1,u1)

def _f0(u0):
    return g(-1,u0)

def _f1(u1):
   return g(1,u1)
 
def first_true( left_bound, right_bound, condition ):
    if right_bound - left_bound <= 0.00001:
        return right_bound
    mid = ( left_bound + right_bound )/2
    if condition(mid):
        return first_true( left_bound, mid, condition )
    else:
        return first_true( mid, right_bound, condition )

def intersection(f, g, left_bound, right_bound): #assumes there is only one
    sgn = lambda x: 1 if f(x) - g(x) >= 0 else -1
    sgn0 = sgn(left_bound)
    con = lambda x: sgn(x) * sgn0 < 0 
    return first_true( left_bound, right_bound, con )

def _n0_mean(u0):
    line = lambda x: (u0-x)/a00
    x = intersection( f0, line, u0_min, u0_max )
    return line(x)

def _n1_mean(u1):
    line = lambda x: (u1-x)/a11
    x = intersection( f1, line, u1_min, u1_max )
    return line(x)


polylog32 = calc(lambda x: _polylog(3/2, x), 0, 1, 'polylog32.csv')
polylog12 = calc(lambda x: _polylog(1/2, x), 0, 1, 'polylog12.csv')

def polylog(s, x):
    if s == 1/2:
        return polylog12(x)
    if s == 3/2:
        return polylog32(x)

f0 = calc(_f0, u0_min, u0_max, 'f0.csv')
f1 = calc(_f1, u1_min, u1_max-0.0001, 'f1.csv')
logz0 = calc(logz0, -10, u0_max, 'logz0.csv')
logz1 = calc(logz1, -10, u1_max, 'logz1.csv')
n0_mean = calc(_n0_mean, -90, 90, 'n0_mean.csv')
n1_mean = calc(_n1_mean, -90, 90, 'n1_mean.csv')
 
def plot(ax, filename):
    df = pd.read_csv(filename, index_col=0)
    ax.plot(df.index.values, df['y']) 
 
def show(filename):
    df = pd.read_csv(filename, index_col=0)
    plt.plot(df.index.values, df['y']) 
    plt.show()


invert('n0_mean.csv')

names = ['polylog32', 'polylog12', 'f0', 'f1', 'logz0', 'logz1', 'n0_mean', 'n1_mean', 'inverse_n0_mean']
#fig, axs = plt.subplots(len(filenames),1)
if(do):
    for name in names:
        fig, ax = plt.subplots()
        plot(ax, name + '.csv')
        fig.savefig(name + '.pdf')
#fig.savefig('wykresy.pdf')
#show('n1_mean.csv')

def show_mean0(u0):
    xs = np.linspace(u0_min, u0_max, 1000)
    ys = [f0(x) for x in xs]
    plt.plot(xs, ys)
    line = lambda x: (u0-x)/a00
    ys = [line(x) for x in xs]
    plt.plot(xs, ys)
    x = intersection(f0, line, u0_min, u0_max)
    y = line(x)
    plt.plot([x], [y], 'o')
    plt.show()

def show_mean1(u1):
    xs = np.linspace(u1_min, u1_max, 1000)
    ys = [f1(x) for x in xs]
    plt.plot(xs, ys)
    line = lambda x: (u1-x)/a11
    ys = [line(x) for x in xs]
    plt.plot(xs, ys)
    x = intersection(f1, line, u1_min, u1_max)
    y = line(x)
    plt.plot([x], [y], 'o')
    plt.show()

