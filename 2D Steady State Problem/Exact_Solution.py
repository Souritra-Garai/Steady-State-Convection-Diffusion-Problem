import numpy as np
import Crank_Nicolson_Method

# Density
rho = 1.0
# Velocity
u = 1.0
# Diffusion coefficient
Gamma = 0.1
# Length of Domain
L = 1.0

# Number of points in grid (minus 1)
m = 1000
# Grid distance
Dx = L / m

# Total time for boundary information 
# to travel across the domain
total_t = 2 # max(abs(rho * L**2 / Gamma), abs(L / u))
# Time step
Dt = 0.0001
# Number of time steps to run (minus 1)
m_t = int(np.ceil(total_t / Dt))

# Courant number
c = u * Dt / Dx
# Ratio to time step to diffusion time across cell
d = Gamma * Dt / (rho * (Dx**2))

phi = Crank_Nicolson_Method.solve(m, Dx, m_t, Dt, d, c, 3)

np.save('Exact_Solution', phi)

