import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.special import zeta
from numpy.linalg import inv
from numpy import array
from numpy import sqrt
from numpy import pi
from matplotlib import pyplot as plt
from calc import _import
import calc

# 0 -- minus   1 -- plus

d = 3
#a = np.array( [[1,0.2],[0.2,1]] )
#a = np.array( [[0.1,0.5],[0.5,0.1]] )
a = calc.a
a00, a01, a11 = a[0][0], a[0][1], a[1][1]
a10 = a01
v = 1/10

u0_max = -v**2
u1_max = 0
u0_min = -100
u1_min = -100

def polylog(s, x):
    if s == 1/2:
        return polylog12(x)
    if s == 3/2:
        return polylog32(x)

def first_true( left_bound, right_bound, condition ):
    if right_bound - left_bound <= 0.00001:
        return right_bound
    mid = ( left_bound + right_bound )/2
    if condition(mid):
        return first_true( left_bound, mid, condition )
    else:
        return first_true( mid, right_bound, condition )

def intersections(f, g, left_bound, right_bound):
    xs = np.linspace(left_bound, right_bound, 1000)
    result_is = []
    sgn = lambda i: 1 if f(xs[i]) - g(xs[i]) >= 0 else -1
    for i in range(1,len(xs)):
        if sgn(i) * sgn(i-1) < 0: #sign changes
            result_is.append(i)
    result = [ intersection(f,g,xs[i-1],xs[i]) for i in result_is ] 
    return result

def intersection(f, g, left_bound, right_bound): #assumes there is only one
    sgn = lambda x: 1 if f(x) - g(x) >= 0 else -1
    sgn0 = sgn(left_bound)
    con = lambda x: sgn(x) * sgn0 < 0 
    return first_true( left_bound, right_bound, con )

logz0 = _import('logz0.csv')
logz1 = _import('logz1.csv')



def F(u0, u1, n0, n1):
    n = np.array([n0,n1])
    u = np.array([u0,u1])
    x = u - a @ n
    x0, x1 = x[0], x[1]
    if x0 > u0_max + 0.001:
        return np.inf
    if x1 > u1_max + 0.001:
        return np.inf
    return - logz0(x0) - logz1(x1) - 1/2 * n @ a @ n

def show_main(u0, u1):
    fig, ax = plt.subplots()
    plot_main(ax, u0, u1)
    plt.show()

def plot_main(ax, u0, u1):
    ax.grid()
    ax.set_ylim([0, 7])
    ax.set_xlim([0, 7])
    left_bound = -1
    right_bound = 10
    f = lambda n0: n1_main(u1, n0)
    g = lambda n0: inverse_n0_main(u0, n0)
    n0s = np.linspace(left_bound, right_bound, 1000)
    ys = [ f(n0) for n0 in n0s ]
    ax.plot(n0s, ys, label='$g_+(n_-)$')
    ys = [ g(n0) for n0 in n0s ]
    ax.plot(n0s, ys, label='$g_-(n_+)$')
    h = lambda n0: F(u0, u1, n0, n1_main(u1, n0))
    #ys = [ h(n0) for n0 in n0s ] 
    #ax.plot(n0s, ys)
    xs = intersections( f, g, left_bound, right_bound ) 
    soli = min( range(len(xs)), key = lambda i: h(xs[i]) )
    ys = [ f(x) for x in xs ]
    xs_sol = [xs[soli]] 
    ys_sol = [ys[soli]]
    xs_rest = xs[:soli] + xs[soli+1:]
    ys_rest = ys[:soli] + ys[soli+1:]
    ax.plot(xs_sol, ys_sol, 'o')
    ax.plot(xs_rest, ys_rest, 'o')
    ax.set_xlabel('$n_-$')
    ax.set_ylabel('$n_+$')
    ax.legend()
    #n1s = np.linspace(0,10, 100)
    #xs = [n0_main(u0, n1) for n1 in n1s]
    #plt.plot(xs, n1s, 'o')
    print(ys[soli] > f1(u1_max), is_condensate(u0,u1))

def is_condensate(u0, u1):
    left_bound = -1
    right_bound = 20
    f = lambda n0: n1_main(u1, n0)
    g = lambda n0: inverse_n0_main(u0, n0)
    xs = intersections( f, g, left_bound, right_bound ) 
    h = lambda n0: F(u0, u1, n0, n1_main(u1, n0))
    soli = min( range(len(xs)), key = lambda i: h(xs[i]) )
    #soli = 0
    ys = [ f(x) for x in xs ]
    return ys[soli] > f1(u1_max)

def densities(u0, u1):
    left_bound = -1
    right_bound = 20
    f = lambda n0: n1_main(u1, n0)
    g = lambda n0: inverse_n0_main(u0, n0)
    xs = intersections( f, g, left_bound, right_bound ) 
    h = lambda n0: F(u0, u1, n0, n1_main(u1, n0))
    soli = min( range(len(xs)), key = lambda i: h(xs[i]) )
    ys = [ f(x) for x in xs ]
    return xs[soli], ys[soli]

def multisol(u0, u1):
    left_bound = -1
    right_bound = 20
    f = lambda n0: n1_main(u1, n0)
    g = lambda n0: inverse_n0_main(u0, n0)
    xs = intersections( f, g, left_bound, right_bound ) 
    return len(xs) > 1

def show_edge():
    fig, ax = plt.subplots()
    ax.grid()
    plot_edge(ax)
    plt.show()

def plot_edge( ax ): 
    u0s = np.linspace(-0.5, 0.9, 32)
    u1s = []
    u1_multisols = []
    u0_multisols = []
    u1_multisol2s = []
    for u0 in u0s:
        con = lambda u1: is_condensate(u0, u1)
        u1 = first_true(0, 2.5, con)
        u1s.append(u1)
        print(u0, u1)
        con = lambda u1: multisol(u0, u1)
        u1_multisol = first_true(0, u1, con)
        if con(u1_multisol):
            u1_multisols.append(u1_multisol)
            u0_multisols.append(u0)
            #con = lambda u1: not con(u1)
            #u1_multisol2 = first_true(u1, right_bound, con)
            #u1_miltisol
    ax.set_xlim([u1s[0], u1s[-1]])
    ax.set_ylim([u0s[0], u0s[-1]])
    ax.grid()
    ax.set_xlabel('$\mu_+$')
    ax.set_ylabel('$\mu_-$')
    ax.text(0.7, 0, 'Kondensat', fontsize=16)
    #plt.title('Diagram Fazowy')
    ax.plot(u1s, u0s)
    #xs = [u1_multisols[0], u1s[-1], u1s[-1], u1_multisols[-1]]
    #ys = [u0_multisols[0], u0_multisols[0], u0_multisols[-1], u0_multisols[-1]]
    #ax.fill(xs, ys, 'orange')
    ax.plot(u1_multisols, u0_multisols)
    return [ u0s, u1s ]

def polylog(s, x):
    if s == 1/2:
        return polylog12(x)
    if s == 3/2:
        return polylog32(x)

f1 = _import('f1.csv')
n0_mean = _import('n0_mean.csv')
inverse_n0_mean = _import('inverse_n0_mean.csv')
n1_mean = _import('n1_mean.csv')

fig, ax = plt.subplots()
xs = np.linspace(0,7,1000)
ax.set_xlim([0, 7])
ax.set_ylim([0, 7])
u0 = 0.6
ys = [ n1_mean( u0 - a10 * x ) for x in xs ]
ax.plot( xs, ys )
fig.savefig('g0.pdf')
plt.show()

def n0_main(u0, n1):
    return n0_mean(u0 - a10*n1)

def n1_main(u1, n0):
    return n1_mean(u1 - a10*n0)

def inverse_n0_main(u0, n0):
    return ( u0 - inverse_n0_mean(n0) ) / a10


show_main(0.6, 0.55)
#show_main(-0.5, 1.5)
#u0 = -0.5
#u1s = np.linspace(0.2, 0.5, 200)
#for u1 in u1s:
#    n0, n1 = densities(u0, u1)
#    print(u0, u1, n0, n1)
#fig, ax = plt.subplots()
#plot_main(ax, 0.6, 0.6)
#fig.savefig('main0606.pdf')

#show_main(0.56,1)
#show_edge()
#fig, ax = plt.subplots()
#u0s, u1s = plot_edge(ax)
#print(u1s)
#ret = 'mu_-, mu_+\n'
#for i in range(len(u0s)):
#    ret += str(u0s[i]) + ' ' +  str(u1s[i]) + '\n'
#with open("brzeg.txt", "w") as text_file:
#    text_file.write(ret)
#tab = [[0.335, -0.5], [0.375, -0.3], [0.415, -0.1], 
#[0.4565, 0.1], [0.5,0.3], [0.535, 0.5], [0.577, 0.7], [0.616, 0.9], [0.66, 1.1], [0.697,1.3], [0.738, 1.5]]
#xs = [ e[0] for e in tab ]
#ys = [ e[1] for e in tab ]
#ax.plot(xs, ys)
#fig.savefig('brzeg.pdf')
