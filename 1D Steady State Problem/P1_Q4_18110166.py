# Written by Souritra Garai
# Date - 8th September 2020
import numpy as np
import matplotlib.pyplot as plt
import P1_Exact_Solution as exact

from P1_Q1_18110166 import upwind_diff_solution
from P1_Q2_18110166 import central_diff_solution

Pe = 50
Pe2 = 18
n = 40

phi_upwind = upwind_diff_solution(n, Pe)
phi_central = central_diff_solution(n, Pe)

x = np.linspace(0, 1, n+1)
Dx = 1 / n
phi = phi_upwind

table = np.zeros((n+1, 5))
table[:, 0] = np.linspace(0, n, n+1)
table[:, 1] = x
table[:, 2] = phi
table[1:-1,3] = ( phi[2:] - phi[:-2] ) / (2*Dx)
table[1:-1,4] = ( phi[2:] - 2*phi[1:-1] + phi[:-2] ) / (Dx**2)
head = ['i', 'x_i', 'phi_i', 'phi_i\'', 'phi_i\'\'']
np.savetxt('P1_Q4_Upwind.csv', delimiter=',', X=table, header=','.join(head))

phi = phi_central

table = np.zeros((n+1, 5))
table[:, 0] = np.linspace(0, n, n+1)
table[:, 1] = x
table[:, 2] = phi
table[1:-1,3] = ( phi[2:] - phi[:-2] ) / (2*Dx)
table[1:-1,4] = ( phi[2:] - 2*phi[1:-1] + phi[:-2] ) / (Dx**2)
head = ['i', 'x_i', 'phi_i', 'phi_i\'', 'phi_i\'\'']
np.savetxt('P1_Q4_Central.csv', delimiter=',', X=table, header=','.join(head))

plt.plot(x, phi_upwind, label='Numerical Solution')

x_new = np.linspace(0, 1, 1000)
plt.plot(x_new, exact.phi(x_new, Pe), label=r'Exact Solution with $Pe=50$')
plt.plot(x_new, exact.phi(x_new, Pe2), label=r'Exact Solution with $Pe=18$')

plt.ylim(-0.5, 1.1)
plt.legend()
plt.xticks(np.arange(0, 1.1, 0.1))
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
plt.xticks(np.arange(0, 1.1, 0.1))
plt.grid()
plt.xlabel(r'$x$', fontdict={'size':14})
plt.ylabel(r'$\varphi (x)$', fontdict={'size':14})
plt.title('Numerical Solution using\nCentral Difference for Convection Term', fontdict={'size':20})

plt.show()