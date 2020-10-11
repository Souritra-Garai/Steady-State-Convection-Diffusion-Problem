# Written by Souritra Garai
# Date - 8th September 2020
import numpy as np
import matplotlib.pyplot as plt
import P1_Exact_Solution as exact

# from TDMA_Solver import solver
from P1_Q1_18110166 import upwind_diff_solution
from P1_Q2_18110166 import central_diff_solution

Pe = 50
Pe2 = 18
n = 10

phi_upwind = upwind_diff_solution(n, Pe)
phi_central = central_diff_solution(n, Pe)

x = np.linspace(0, 1, n+1)

plt.plot(x, phi_upwind, label='Numerical Solution')

x_new = np.linspace(0, 1, 1000)
plt.plot(x_new, exact.phi(x_new, Pe), label=r'Exact Solution with $Pe=50$')
plt.plot(x_new, exact.phi(x_new, Pe2), label=r'Exact Solution with $Pe=18$')

plt.ylim(-0.5, 1.1)
plt.legend()
plt.xticks(np.arange(0, 1 + 1/n, 1/n))
plt.grid()
plt.xlabel(r'$x$', fontdict={'size':14})
plt.ylabel(r'$\varphi (x)$', fontdict={'size':14})
plt.title('Numerical Solution using\nUpwind Difference for Convection Term', fontdict={'size':20})

plt.show()

plt.plot(x, phi_central, label='Numerical Solution')

x_new = np.linspace(0, 1, 1000)
plt.plot(x_new, exact.phi(x_new, Pe), label=r'Exact Solution with $Pe=50$')

plt.ylim(-0.5, 1.1)
plt.legend()
plt.xticks(np.arange(0, 1 + 1/n, 1/n))
plt.grid()
plt.xlabel(r'$x$', fontdict={'size':14})
plt.ylabel(r'$\varphi (x)$', fontdict={'size':14})
plt.title('Numerical Solution using\nCentral Difference for Convection Term', fontdict={'size':20})

plt.show()