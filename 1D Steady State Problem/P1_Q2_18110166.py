# Written by Souritra Garai
# Date - 8th September 2020
import numpy as np
import P1_Exact_Solution as exact

from TDMA_Solver import solver

Pe = 50
n = 10

def central_diff_solution(n, Pe) :

    x_min, x_max = 0.0, 1.0

    Dx = (x_max - x_min) / n

    delta = 2 * ( Dx**2 )
    lamda = 2 * Dx

    e = 2*lamda + Pe*delta
    f =-4*lamda
    g = 2*lamda - Pe*delta

    # x = np.linspace(x_min, x_max, n+1)

    phi = np.zeros(n+1)
    phi[0] = 0
    phi[-1] = 1

    E = np.zeros(n-1) + e
    F = np.zeros(n-1) + f
    G = np.zeros(n-1) + g

    matrix_eqn_solver = solver.generate_solver(E, F, G)

    b = np.zeros(n-1)
    b[0] = - E[0] * phi[0]
    b[-1] = - G[-1] * phi[-1]

    phi[1:-1] = matrix_eqn_solver.solve(b)

    return phi

x = np.linspace(0, 1, n+1)
Dx = 1 / n
phi = central_diff_solution(n, Pe)

table = np.zeros((n+1, 5))
table[:, 0] = np.linspace(0, n, n+1)
table[:, 1] = x
table[:, 2] = phi
table[1:-1,3] = ( phi[2:] - phi[:-2] ) / (2*Dx)
table[1:-1,4] = ( phi[2:] - 2*phi[1:-1] + phi[:-2] ) / (Dx**2)
head = ['i', 'x_i', 'phi_i', 'phi_i\'', 'phi_i\'\'']
np.savetxt('P1_Q2.csv', delimiter=',', X=table, header=','.join(head))
