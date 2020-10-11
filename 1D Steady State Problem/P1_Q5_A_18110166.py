# Written by Souritra Garai
# Date - 13th September 2020
import numpy as np
import matplotlib.pyplot as plt
import P1_Exact_Solution as exact

from TDMA_Solver import solver

Pe = 50
x_max, x_min = 1, 0

n_array = [10, 10, 10, 100, 100, 100]
r_array = [ 0.9, 0.7, 0.5, 0.90, 0.95, 0.99]
k = 1
for n, r in zip(n_array, r_array) :

    a = (x_max - x_min) * (1 - r) / (1 - r**n)
    Dx = a * r**(np.linspace(0, n-1, n))
    x = np.append(0, np.cumsum(Dx))

    delta = Dx[1:]**2 + Dx[:-1]**2
    gamma = Dx[1:] - Dx[:-1]

    E = 2*Dx[:-1] - 2*gamma + Pe*delta
    F = 2*gamma - 4*Dx[:-1] - Pe*delta
    G = 2*Dx[:-1]

    matrix_eqn_solver = solver.generate_solver(E, F, G)

    phi = np.zeros(n+1)
    phi[0] = 0
    phi[n] = 1

    b = np.zeros(n-1)
    b[0] = - E[0]*phi[0]
    b[-1] = - G[-1]*phi[-1]

    phi[1:-1] = matrix_eqn_solver.solve(b)

    if n == 10 :
        
        plt.plot(x, phi, label='Numerical Solution', lw=3, marker='X')

    else :

        plt.plot(x, phi, label='Numerical Solution', lw=3)

    x_new = np.linspace(0, 1, 1000)
    plt.plot(x_new, exact.phi(x_new, Pe), label=r'Exact Solution with $Pe=50$', ls='-.')

    plt.legend()
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.grid()
    plt.xlabel(r'$x$', fontdict={'size':14})
    plt.ylabel(r'$\varphi (x)$', fontdict={'size':14})
    plt.title('$n='+str(n)+'$\n$r='+str(r)+'$', fontdict={'size':15})

    plt.savefig('P1_Q5_A_F'+str(k)+'.png')
    k += 1
    plt.show()

n_array = [10, 10, 10, 100, 100, 100]
r_array = [ 0.9, 0.8, 0.5, 0.90, 0.95, 0.99]

for n, r in zip(n_array, r_array) :

    a = (x_max - x_min) * (1 - r) / (1 - r**n)
    Dx = a * r**(np.linspace(0, n-1, n))
    x = np.append(0, np.cumsum(Dx))

    delta = Dx[1:]**2 + Dx[:-1]**2
    gamma = Dx[1:] - Dx[:-1]
    lamda = Dx[1:] + Dx[:-1]

    E = 2*lamda - 2*gamma + Pe*delta
    F = - 4*lamda
    G = 2*lamda + 2*gamma - Pe*delta

    matrix_eqn_solver = solver.generate_solver(E, F, G)

    phi = np.zeros(n+1)
    phi[0] = 0
    phi[n] = 1

    b = np.zeros(n-1)
    b[0] = - E[0]*phi[0]
    b[-1] = - G[-1]*phi[-1]

    phi[1:-1] = matrix_eqn_solver.solve(b)

    if n == 10 :
        
        plt.plot(x, phi, label='Numerical Solution', lw=3, marker='X')

    else :

        plt.plot(x, phi, label='Numerical Solution', lw=3)

    x_new = np.linspace(0, 1, 1000)
    plt.plot(x_new, exact.phi(x_new, Pe), label=r'Exact Solution with $Pe=50$', ls='-.')

    plt.legend()
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.grid()
    plt.xlabel(r'$x$', fontdict={'size':14})
    plt.ylabel(r'$\varphi (x)$', fontdict={'size':14})
    plt.title('$n='+str(n)+'$\n$r='+str(r)+'$', fontdict={'size':15})

    plt.savefig('P1_Q5_A_F'+str(k)+'.png')
    k += 1
    plt.show()



