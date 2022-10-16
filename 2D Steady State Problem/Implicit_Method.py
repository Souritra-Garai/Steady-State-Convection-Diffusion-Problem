import numpy as np
import PDMA_Solver
import TDMA_Solver
import Coefficients

def solve_3rd_Order_UDS(m, Dx, m_t, Dt, d, c) :

    # phi matrix
    phi = np.zeros((m_t+1, m+1), dtype=float)
    # Initial condition
    phi[0, :] = 0
    # Boundary Conditions
    phi[:, 0] = 0.0
    phi[:,-1] = 1.0

    # Creating the banded matrix
    D = np.zeros(m-1)
    E = np.zeros(m-1)
    F = np.zeros(m-1)
    G = np.zeros(m-1)
    H = np.zeros(m-1)

    higher_order_coeffs = Coefficients.Coefficients_3rd_Order_Implicit_Scheme(d, c)
    lower_order_coeffs  = Coefficients.Coefficients_1st_Order_Implicit_Scheme(d, c)

    for i, arr in enumerate([H, G, F, E, D]) :

        arr[1:-1]   = higher_order_coeffs[i]
        arr[0]      = lower_order_coeffs[i]
        arr[-1]     = lower_order_coeffs[i]

    # exit()
    my_solver = PDMA_Solver.solver.generate_solver(D, E, F, G, H)

    # b array for matrix equations
    b = np.zeros(m-1)

    # Solving for phi
    # while marching forward in time
    for n in range(1, m_t + 1) :

        b[:] = phi[n-1, 1:-1]
        
        b[0]    -= E[0] * phi[n, 0]
        b[1]    -= D[1] * phi[n, 0]
        b[-1]   -= G[-1] * phi[n,-1]
        b[-2]   -= H[-2] * phi[n,-1]

        phi[n, 1:-1] = my_solver.solve(b)

    return phi

def solve_2nd_Order_CDS(m, Dx, m_t, Dt, d, c) :

    # phi matrix
    phi = np.zeros((m_t+1, m+1), dtype=float)
    # Initial condition
    phi[0, :] = 0
    # Boundary Conditions
    phi[:, 0] = 0.0
    phi[:,-1] = 1.0

    # Creating the banded matrix
    E = np.zeros(m-1)
    F = np.zeros(m-1)
    G = np.zeros(m-1)

    lower_order_coeffs  = Coefficients.Coefficients_2nd_Order_Implicit_Scheme(d, c)

    E[:] = lower_order_coeffs[3]
    F[:] = lower_order_coeffs[2]
    G[:] = lower_order_coeffs[1]

    # exit()
    my_solver = TDMA_Solver.solver.generate_solver(E, F, G)

    # b array for matrix equations
    b = np.zeros(m-1)

    # Solving for phi
    # while marching forward in time
    for n in range(1, m_t + 1) :

        b[:] = phi[n-1, 1:-1]
        
        b[0]    -= E[0] * phi[n, 0]
        b[-1]   -= G[-1] * phi[n,-1]

        phi[n, 1:-1] = my_solver.solve(b)

    return phi

def solve_1st_Order_UDS(m, Dx, m_t, Dt, d, c) :

    # phi matrix
    phi = np.zeros((m_t+1, m+1), dtype=float)
    # Initial condition
    phi[0, :] = 0
    # Boundary Conditions
    phi[:, 0] = 0.0
    phi[:,-1] = 1.0

    # Creating the banded matrix
    E = np.zeros(m-1)
    F = np.zeros(m-1)
    G = np.zeros(m-1)

    lower_order_coeffs  = Coefficients.Coefficients_1st_Order_Implicit_Scheme(d, c)

    E[:] = lower_order_coeffs[3]
    F[:] = lower_order_coeffs[2]
    G[:] = lower_order_coeffs[1]

    # exit()
    my_solver = TDMA_Solver.solver.generate_solver(E, F, G)

    # b array for matrix equations
    b = np.zeros(m-1)

    # Solving for phi
    # while marching forward in time
    for n in range(1, m_t + 1) :

        b[:] = phi[n-1, 1:-1]
        
        b[0]    -= E[0] * phi[n, 0]
        b[-1]   -= G[-1] * phi[n,-1]

        phi[n, 1:-1] = my_solver.solve(b)

    return phi

def solve(m, Dx, m_t, Dt, d, c, order=3) :

    if order == 1 :

        phi = solve_1st_Order_UDS(m, Dx, m_t, Dt, d, c)

    elif order == 2 :

        phi = solve_2nd_Order_CDS(m, Dx, m_t, Dt, d, c)

    elif order == 3 :

        phi = solve_3rd_Order_UDS(m, Dx, m_t, Dt, d, c)

    else :

        raise ValueError('Only order 1, 2, 3 are defined!!')

    return phi

if __name__ == "__main__":
    
    # Density
    rho = 1.0
    # Velocity
    u = - 1.0
    # Diffusion coefficient
    Gamma = 0.001
    # Length of Domain
    L = 1.0

    # Number of points in grid (minus 1)
    m = 50
    # Position of grid points
    x = np.linspace(0, L, m+1)
    # Grid distance
    Dx = L / m

    # Total time for boundary information 
    # to travel across the domain
    total_t = 1 # max(rho* L**2 / Gamma, L / u)
    print('Time Scale Required :', total_t)
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

    phi = solve(m, Dx, m_t, Dt, d, c, 1)

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Plotting the solution
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Axes
    X, Y = np.meshgrid(t, x, indexing='ij') 

    # Surface Plot
    surf = ax.plot_surface(X, Y, phi, cmap='magma')

    # Axes labels
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.set_zlabel(r'$\varphi$')

    # Colour Bar
    fig.colorbar(surf)

    plt.show()


