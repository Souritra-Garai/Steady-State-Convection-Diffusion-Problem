# Written by Souritra Garai
# Date - 13th September 2020
import numpy as np
import matplotlib.pyplot as plt
import P1_Exact_Solution as exact

from P1_Q1_18110166 import upwind_diff_solution
from LUDecomposition_solver import SolveLUDecomposition

Pe = 50
n_array = [10, 20, 100]
x_min, x_max = 0, 1

def higher_order_solution(n, Pe) :

    Dx = (x_max - x_min) / n

    delta =   4*Pe*Dx - 16
    gamma =   6*Pe*Dx + 30
    lamda = -12*Pe*Dx - 16
    mu    =   2*Pe*Dx + 1

    A = np.zeros((n-1, n-1))
    b = np.zeros(n-1)

    A[0, 0:2] = np.array([-(2+Pe*Dx), 1])
    A[-1, -2:] = np.array([1+Pe*Dx, -(2+Pe*Dx)])

    A[1, 0:4] = np.array([lamda, gamma, delta, 1])
    A[-2, -4:] = np.array([mu, lamda, gamma, delta])

    for i in range(2, n-3) :

        A[i, i-2:i+3] = np.array([mu, lamda, gamma, delta, 1])

    b[-2:] = np.array([-1, -1])

    phi = np.zeros([n+1])
    phi[-1] = 1

    phi[1:-1] = SolveLUDecomposition(A, b, n-1)

    return phi

for n in n_array :

    x = np.linspace(x_min, x_max, n+1)
    phi = higher_order_solution(n, Pe)

    plt.plot(x, phi, label='Number of Grid Points = '+str(n+1), lw=1)

plt.plot(x, exact.phi(x, Pe), label='Exact Solution', ls='-.', lw=1)

plt.legend()
plt.xticks(np.arange(0, 1.1, 0.1))
plt.grid()
plt.xlabel(r'$x$', fontdict={'size':14})
plt.ylabel(r'$\varphi (x)$', fontdict={'size':14})
plt.title('Numerical Solution using\nHigher Order Schemes', fontdict={'size':20})
plt.show()

for n in n_array :

    x = np.linspace(x_min, x_max, n+1)
    phi = upwind_diff_solution(n, Pe)

    plt.plot(x, phi, label='Number of Grid Points = '+str(n+1), lw=1)

plt.plot(x, exact.phi(x, Pe), label='Exact Solution', ls='-.', lw=1)

plt.legend()
plt.xticks(np.arange(0, 1.1, 0.1))
plt.grid()
plt.xlabel(r'$x$', fontdict={'size':14})
plt.ylabel(r'$\varphi (x)$', fontdict={'size':14})
plt.title('Numerical Solution using\nLower Order Schemes', fontdict={'size':20})
plt.show()