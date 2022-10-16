import numpy as np
import PDMA_Solver
import TDMA_Solver
import Coefficients

def solve_3rd_Order_UDS_RK(m, Dx, m_t, Dt, d, c) :

    # phi matrix
    phi = np.zeros((m_t+1, m+1), dtype=float)
    # Initial condition
    phi[0, :] = 0
    # Boundary Conditions
    phi[:, 0] = 0.0
    phi[:,-1] = 1.0

    Higher_Order_Coeffs = Coefficients.Coefficients_3rd_Order_Explicit_Scheme(d, c)
    Lower_Order_Coeffs  = Coefficients.Coefficients_1st_Order_Explicit_Scheme(d, c)

    def f(phi) :

        p1 = Coefficients.multiply(phi[ :3], Lower_Order_Coeffs)
        p3 = Coefficients.multiply(phi[-3:], Lower_Order_Coeffs)

        p2 = Coefficients.multiply( np.column_stack( ( phi[4:], phi[3:-1], phi[2:-2], phi[1:-3], phi[:-4] ) ), Higher_Order_Coeffs)

        return np.hstack((p1, p2, p3))

    # Arrays for storing midpoint values 
    # in 4th order Runge Kutta method
    p1 = np.zeros(m+1)
    p2 = np.zeros(m+1)
    p3 = np.zeros(m+1)
    # From Boundary condition
    p1[-1] = p2[-1] = p3[-1] = 1.0

    # Solving for phi
    # while marching along forward time
    for n in range(m_t) :

        # Using 4th order Runge-Kutta
        k1 = f(phi[n, :])
        p1[1:-1] = phi[n, 1:-1] + 0.5 * k1

        k2 = f(p1)
        p2[1:-1] = phi[n, 1:-1] + 0.5 * k1

        k3 = f(p2)
        p3[1:-1] = phi[n, 1:-1] + k3

        k4 = f(p3)

        phi[n+1, 1:-1] = phi[n, 1:-1] + (k1 + 2*k2 + 2*k3 + k4) / 6

    return phi

def solve_2nd_Order_CDS_RK(m, Dx, m_t, Dt, d, c) :

    # phi matrix
    phi = np.zeros((m_t+1, m+1), dtype=float)
    # Initial condition
    phi[0, :] = 0
    # Boundary Conditions
    phi[:, 0] = 0.0
    phi[:,-1] = 1.0

    coeffs  = Coefficients.Coefficients_2nd_Order_Explicit_Scheme(d, c)

    def f(phi) :

        return Coefficients.multiply( np.column_stack( ( phi[2:], phi[1:-1], phi[:-2] ) ), coeffs )

    # Arrays for storing midpoint values 
    # in 4th order Runge Kutta method
    p1 = np.zeros(m+1)
    p2 = np.zeros(m+1)
    p3 = np.zeros(m+1)
    # From Boundary condition
    p1[-1] = p2[-1] = p3[-1] = 1.0

    # Solving for phi
    # while marching along forward time
    for n in range(m_t) :

        # Using 4th order Runge-Kutta
        k1 = f(phi[n, :])
        p1[1:-1] = phi[n, 1:-1] + 0.5 * k1

        k2 = f(p1)
        p2[1:-1] = phi[n, 1:-1] + 0.5 * k1

        k3 = f(p2)
        p3[1:-1] = phi[n, 1:-1] + k3

        k4 = f(p3)

        phi[n+1, 1:-1] = phi[n, 1:-1] + (k1 + 2*k2 + 2*k3 + k4) / 6

    return phi

def solve_1st_Order_UDS_RK(m, Dx, m_t, Dt, d, c) :

    # phi matrix
    phi = np.zeros((m_t+1, m+1), dtype=float)
    # Initial condition
    phi[0, :] = 0
    # Boundary Conditions
    phi[:, 0] = 0.0
    phi[:,-1] = 1.0

    coeffs  = Coefficients.Coefficients_1st_Order_Explicit_Scheme(d, c)

    def f(phi) :

        return Coefficients.multiply( np.column_stack( ( phi[2:], phi[1:-1], phi[:-2] ) ), coeffs )

    # Arrays for storing midpoint values 
    # in 4th order Runge Kutta method
    p1 = np.zeros(m+1)
    p2 = np.zeros(m+1)
    p3 = np.zeros(m+1)
    # From Boundary condition
    p1[-1] = p2[-1] = p3[-1] = 1.0

    # Solving for phi
    # while marching along forward time
    for n in range(m_t) :

        # Using 4th order Runge-Kutta
        k1 = f(phi[n, :])
        p1[1:-1] = phi[n, 1:-1] + 0.5 * k1

        k2 = f(p1)
        p2[1:-1] = phi[n, 1:-1] + 0.5 * k1

        k3 = f(p2)
        p3[1:-1] = phi[n, 1:-1] + k3

        k4 = f(p3)

        phi[n+1, 1:-1] = phi[n, 1:-1] + (k1 + 2*k2 + 2*k3 + k4) / 6

    return phi

def solve_3rd_Order_UDS(m, Dx, m_t, Dt, d, c) :

    # phi matrix
    phi = np.zeros((m_t+1, m+1), dtype=float)
    # Initial condition
    phi[0, :] = 0
    # Boundary Conditions
    phi[:, 0] = 0.0
    phi[:,-1] = 1.0

    Higher_Order_Coeffs = Coefficients.Coefficients_3rd_Order_Explicit_Scheme(d, c)
    Lower_Order_Coeffs  = Coefficients.Coefficients_1st_Order_Explicit_Scheme(d, c)

    def f(phi) :

        p1 = Coefficients.multiply(phi[ :3], Lower_Order_Coeffs)
        p3 = Coefficients.multiply(phi[-3:], Lower_Order_Coeffs)

        p2 = Coefficients.multiply( np.column_stack( ( phi[4:], phi[3:-1], phi[2:-2], phi[1:-3], phi[:-4] ) ), Higher_Order_Coeffs)

        return np.hstack((p1, p2, p3))

    # Euler's Method
    for n in range(m_t) :

        # phi[n+1, 1:-1] = (1 - 2*d) * phi[n, 1:-1] + (d - c/2) * phi[n, 2:] + (d + c/2) * phi[n, :-2]
        phi[n+1, 1:-1] = phi[n, 1:-1] + f(phi[n])

    return phi

def solve_2nd_Order_CDS(m, Dx, m_t, Dt, d, c) :

    # phi matrix
    phi = np.zeros((m_t+1, m+1), dtype=float)
    # Initial condition
    phi[0, :] = 0
    # Boundary Conditions
    phi[:, 0] = 0.0
    phi[:,-1] = 1.0

    coeffs  = Coefficients.Coefficients_2nd_Order_Explicit_Scheme(d, c)

    def f(phi) :

        return Coefficients.multiply( np.column_stack( ( phi[2:], phi[1:-1], phi[:-2] ) ), coeffs )

    # Arrays for storing midpoint values 
    # in 4th order Runge Kutta method
    p1 = np.zeros(m+1)
    p2 = np.zeros(m+1)
    p3 = np.zeros(m+1)
    # From Boundary condition
    p1[-1] = p2[-1] = p3[-1] = 1.0

    # Euler's Method
    for n in range(m_t) :

        # phi[n+1, 1:-1] = (1 - 2*d) * phi[n, 1:-1] + (d - c/2) * phi[n, 2:] + (d + c/2) * phi[n, :-2]
        phi[n+1, 1:-1] = phi[n, 1:-1] + f(phi[n])

    return phi

def solve_1st_Order_UDS(m, Dx, m_t, Dt, d, c) :

    # phi matrix
    phi = np.zeros((m_t+1, m+1), dtype=float)
    # Initial condition
    phi[0, :] = 0
    # Boundary Conditions
    phi[:, 0] = 0.0
    phi[:,-1] = 1.0

    coeffs  = Coefficients.Coefficients_1st_Order_Explicit_Scheme(d, c)

    def f(phi) :

        return Coefficients.multiply( np.column_stack( ( phi[2:], phi[1:-1], phi[:-2] ) ), coeffs )

    # Arrays for storing midpoint values 
    # in 4th order Runge Kutta method
    p1 = np.zeros(m+1)
    p2 = np.zeros(m+1)
    p3 = np.zeros(m+1)
    # From Boundary condition
    p1[-1] = p2[-1] = p3[-1] = 1.0

    # Euler's Method
    for n in range(m_t) :

        # phi[n+1, 1:-1] = (1 - 2*d) * phi[n, 1:-1] + (d - c/2) * phi[n, 2:] + (d + c/2) * phi[n, :-2]
        phi[n+1, 1:-1] = phi[n, 1:-1] + f(phi[n])

    return phi

def solve(m, Dx, m_t, Dt, d, c, order=3, Runge_Kutta=True) :

    if order == 1 :

        phi = solve_1st_Order_UDS_RK(m, Dx, m_t, Dt, d, c) if Runge_Kutta else solve_1st_Order_UDS(m, Dx, m_t, Dt, d, c)

    elif order == 2 :

        phi = solve_2nd_Order_CDS_RK(m, Dx, m_t, Dt, d, c) if Runge_Kutta else solve_2nd_Order_CDS(m, Dx, m_t, Dt, d, c)

    elif order == 3 :

        phi = solve_3rd_Order_UDS_RK(m, Dx, m_t, Dt, d, c) if Runge_Kutta else solve_3rd_Order_UDS(m, Dx, m_t, Dt, d, c)

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

    phi = solve(m, Dx, m_t, Dt, d, c, 2, False)

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