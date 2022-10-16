import numpy as np
import Explicit_Method
import Implicit_Method
import Crank_Nicolson_Method
import matplotlib.pyplot as plt

from scipy.interpolate import interp2d

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
u = 1.0
# Diffusion coefficient
Gamma = 0.1
# Length of Domain
L = 1.0

# Total time for boundary information 
# to travel across the domain
total_t = 2

m = 20
x = np.linspace(0, L, m+1)
Dx = L / m

exact_phi = np.load('Exact_Solution.npy')

k = np.linspace(0, exact_phi.shape[1]-1, m+1, dtype=int)

data_array = dict()

for name in method_names :

    data_array[name] = []

for Dt in 10 ** np.linspace(-3, 0, 50) :

    # Number of time steps to run (minus 1)
    m_t = int(np.ceil(total_t / Dt))
    t = np.linspace(0, m_t*Dt, m_t+1)

    l = np.linspace(0, exact_phi.shape[0]-1, m_t+1, dtype=int)

    X, Y = np.meshgrid(l, k, indexing='ij')

    e_phi = exact_phi[X, Y]

    # Peclet Number
    Pe = rho * u * Dx / Gamma
    # Courant number
    c = u * Dt / Dx
    # Ratio to time step to diffusion time across cell
    d = Gamma * Dt / (rho * (Dx**2))

    for solve, name in zip(methods, method_names) :

        phi = solve(m, Dx, m_t, Dt, d, c)

        if np.less_equal(phi, 5).all() :
            
            error = np.average( np.square( e_phi - phi ) )
            data_array[name].append([Dt, error])

        else :
            
            print(name, 'failed!!')

            with open('Compare_Solutions_Log_File.txt', 'a') as log :

                log.write(name + 'failed at' + 'Pe = ' + str(round(Pe, 3)) + ', c = ' + str(round(c, 3)) + ', d = ' + str(round(d, 3)) + r', $\Delta t$ = ' + str(round(Dt,3)) + r', $\Delta x$ =' + str(round(Dx,3)))

for i in data_array.keys() :

    arr = np.array(data_array[i])
    
    if arr.shape[0] > 0 : plt.plot(arr[:, 0], arr[:, 1], label=i, marker='X')

plt.xlabel(r'$\Delta$ t')
plt.ylabel('Mean Square Error')
plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.show()