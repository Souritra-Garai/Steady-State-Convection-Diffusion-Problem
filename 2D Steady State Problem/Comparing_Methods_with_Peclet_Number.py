import time, os
import numpy as np
import Implicit_Method
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import Explicit_Method
import Implicit_Method
import Crank_Nicolson_Method

methods = [
    Explicit_Method.solve_1st_Order_UDS,
    Explicit_Method.solve_2nd_Order_CDS,
    Explicit_Method.solve_3rd_Order_UDS,
    Explicit_Method.solve_1st_Order_UDS_RK,
    Explicit_Method.solve_2nd_Order_CDS_RK,
    Explicit_Method.solve_3rd_Order_UDS_RK,
    Implicit_Method.solve_1st_Order_UDS,
    Implicit_Method.solve_2nd_Order_CDS,
    Implicit_Method.solve_3rd_Order_UDS,
    Crank_Nicolson_Method.solve_1st_Order_UDS,
    Crank_Nicolson_Method.solve_2nd_Order_CDS,
    Crank_Nicolson_Method.solve_3rd_Order_UDS
]

method_names = [
    '1st Order Euler\'s Explicit Method',
    '2nd Order Euler\'s Explicit Method',
    '3rd Order Euler\'s Explicit Method',
    '1st Order Explicit Method using 4th Order Runge-Kutta',
    '2nd Order Explicit Method using 4th Order Runge-Kutta',
    '3rd Order Explicit Method using 4th Order Runge-Kutta',
    '1st Order Euler\'s Implicit Method',
    '2nd Order Euler\'s Implicit Method',
    '3rd Order Euler\'s Implicit Method',
    '1st Order Crank Nicolson Method',
    '2nd Order Crank Nicolson Method',
    '3rd Order Crank Nicolson Method',
]

# Density
rho = 1.0
# Velocity
u = -1.0
# Length of Domain
L = 1.0

for i, Gamma in enumerate([0.1]) :

    print('\nIteration # ', i+1, '\n')

    # Number of points in grid (minus 1)
    m = 50
    # Position of grid points
    x = np.linspace(0, L, m+1)
    # Grid distance
    Dx = L / m

    # Total time for boundary information 
    # to travel across the domain
    total_t = max(abs(rho * L**2 / Gamma), abs(L / u))
    print('Time Scale Required :', total_t, '\n')
    # Time step
    Dt = 0.01
    # Number of time steps to run (minus 1)
    m_t = int(np.ceil(total_t / Dt))
    # Time grid
    t = np.linspace(0, m_t * Dt, m_t + 1, dtype=float)

    # Cell Peclet Number
    Pe = rho * u * Dx / Gamma
    # Courant number
    c = u * Dt / Dx
    # Ratio to time step to diffusion time across cell
    d = Gamma * Dt / (rho * (Dx**2))

    phi = Crank_Nicolson_Method.solve(m, Dx, m_t, Dt, d, c, order=3)

    for n in range(1, m_t) :

        if np.isclose(phi[n], phi[n-1]).all() :
            
            break

    # Plotting the solution
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Axes
    X, Y = np.meshgrid(t[:n], x, indexing='ij') 

    # Surface Plot
    surf = ax.plot_surface(X, Y, phi[:n, :], cmap='magma')

    # Axes labels
    ax.set_xlabel('t', fontdict={'size':15})
    ax.set_ylabel('x', fontdict={'size':15})
    ax.set_zlabel(r'$\varphi$', fontdict={'size':15})

    plt.show()

    ch = input('Gamma = ' + str(Gamma) + '\nProceed ? (y/n)... ')

    if ch == 'y' :

        # Number of points in grid (minus 1)
        m = 1000
        # Position of grid points
        x = np.linspace(0, L, m+1)
        # Grid distance
        Dx = L / m

        # Total time for boundary information 
        # to travel across the domain
        total_t = t[n]
        # Time step
        Dt = 0.0001
        # Number of time steps to run (minus 1)
        m_t = int(np.ceil(total_t / Dt))
        # Time grid
        t = np.linspace(0, m_t * Dt, m_t + 1, dtype=float)

        # Cell Peclet Number
        Pe = rho * u * Dx / Gamma
        # Courant number
        c = u * Dt / Dx
        # Ratio to time step to diffusion time across cell
        d = Gamma * Dt / (rho * (Dx**2))

        exact_phi = Crank_Nicolson_Method.solve(m, Dx, m_t, Dt, d, c, order=3)

        # Plotting the solution
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Axes
        X, Y = np.meshgrid(t, x, indexing='ij') 

        # Surface Plot
        surf = ax.plot_surface(X, Y, exact_phi, cmap='magma')

        # Axes labels
        ax.set_xlabel('t', fontdict={'size':15})
        ax.set_ylabel('x', fontdict={'size':15})
        ax.set_zlabel(r'$\varphi$', fontdict={'size':15})

        ax.set_title(r'$\Gamma$ = ' + str(round(Gamma, 2)), fontdict={'size':15})

        plt.show()

        # Number of points in grid (minus 1)
        m = 20
        # Position of grid points
        x = np.linspace(0, L, m+1)
        # Grid distance
        Dx = L / m

        # Time step
        Dt = 0.01
        # Number of time steps to run (minus 1)
        m_t = int(np.ceil(total_t / Dt))
        # Time grid
        t = np.linspace(0, m_t * Dt, m_t + 1, dtype=float)

        # Cell Peclet Number
        Pe = rho * u * Dx / Gamma
        # Courant number
        c = u * Dt / Dx
        # Ratio to time step to diffusion time across cell
        d = Gamma * Dt / (rho * (Dx**2))

        print('Cell Peclet Number :', Pe)
        print('Courant Number :', c)
        print('d :', d)

        for plot_no in range(4) :

            plt.subplot(2, 2, plot_no + 1)

            for thickness, solve, name in zip(np.linspace(5, 1, 3), methods[plot_no*3:(plot_no+1)*3], method_names[plot_no*3:(plot_no+1)*3]) :

                phi = solve(m, Dx, m_t, Dt, d, c)

                if np.less_equal(np.abs(phi), 5).all() :
                    
                    for i in [2, m // 2, m - 2] :

                        plt.plot(t, phi[:, i], marker='x', label='x = ' + str(round(x[i], 2)) + ' ' + name[:10], lw=thickness)

                else :
                    
                    print(name, 'failed!!\n')

                    with open('Compare_Solutions_Log_File.txt', 'a') as log :

                        log.write(name + ' failed at ' + 'Pe = ' + str(round(Pe, 3)) + ', c = ' + str(round(c, 3)) + ', d = ' + str(round(d, 3)) + r', $\Delta t$ = ' + str(round(Dt,3)) + r', $\Delta x$ =' + str(round(Dx,3)) + '\n')

            k = np.where(np.linspace(0, L, exact_phi.shape[1])==x[-2])[-1]
            l = np.where(np.linspace(0, L, exact_phi.shape[1])==x[2])[-1]

            for i in [l, (l+k) // 2, k] :

                plt.plot(np.linspace(0, total_t, exact_phi.shape[0]), exact_phi[:, i], c='black')

            # plt.ylim([0.8, 1])
            plt.xlabel('t')
            plt.ylabel(r'$\varphi$')
            plt.grid()
            plt.minorticks_on()
            plt.grid(which='minor', ls='--')
            plt.legend(loc=10, framealpha=0.4)
            plt.title(name[10:], fontdict={'size':15})

        plt.suptitle('Pe = ' + str(round(Pe, 3)) + ', c = ' + str(round(c, 3)) + ', d = ' + str(round(d, 3)) + r', $\Delta t$ = ' + str(round(Dt,3)) + r', $\Delta x$ =' + str(round(Dx,3)), fontsize=15 )
        plt.show()


        



        
