# written by - Souritra Garai
# date - 13th September 2020

from sympy import symbols, simplify, collect
from sympy.solvers.solveset import linsolve

a, b, c, d, e = symbols('a, b, c, d, e')

phi_ip2, phi_ip1, phi_i, phi_im1, phi_im2 = symbols('phi_ip2, phi_ip1, phi_i, phi_im1, phi_im2')

Dx = symbols('Dx')

Pe = symbols('Pe')

# Convection term
eqn1 = a - phi_i
eqn2 = a + b*Dx + c*(Dx**2) + d*(Dx**3) - phi_ip1
eqn3 = a - b*Dx + c*(Dx**2) - d*(Dx**3) - phi_im1
eqn4 = a - 2*b*Dx + 4*c*(Dx**2) - 8*d*(Dx**3) - phi_im2

solution, = linsolve([eqn1, eqn2, eqn3, eqn4], a, b, c, d)

first_derivative = simplify(solution[1])

# Diffusion term
eqn1 = a - phi_i
eqn2 = a + b*Dx + c*(Dx**2) + d*(Dx**3) + e*(Dx**4) - phi_ip1
eqn3 = a - b*Dx + c*(Dx**2) - d*(Dx**3) + e*(Dx**4) - phi_im1
eqn4 = a + 2*b*Dx + 4*c*(Dx**2) + 8*d*(Dx**3) + 16*e*(Dx**4) - phi_ip2
eqn5 = a - 2*b*Dx + 4*c*(Dx**2) - 8*d*(Dx**3) + 16*e*(Dx**4) - phi_im2

solution, = linsolve([eqn1, eqn2, eqn3, eqn4, eqn5], a, b, c, d, e)

second_derivative = simplify(2*solution[2])

Discretized_equation = simplify((Pe*first_derivative - second_derivative)*(Dx**2)*12)
print('Discretized equation')
print(Discretized_equation)