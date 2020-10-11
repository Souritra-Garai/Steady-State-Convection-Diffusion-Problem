# written by - Souritra Garai
# date - 13th September 2020

from sympy import symbols, simplify, collect
from sympy.solvers.solveset import linsolve

a, b, c, d = symbols('a, b, c, d')

phi_ip1, phi_i, phi_im1, phi_im2 = symbols('phi_ip1, phi_i, phi_im1, phi_im2')

Dx = symbols('Dx')

eqn1 = a - phi_i
eqn2 = a + b*Dx + c*(Dx**2) + d*(Dx**3) - phi_ip1
eqn3 = a - b*Dx + c*(Dx**2) - d*(Dx**3) - phi_im1
eqn4 = a - 2*b*Dx + 4*c*(Dx**2) - 8*d*(Dx**3) - phi_im2

solution, = linsolve([eqn1, eqn2, eqn3, eqn4], a, b, c, d)

a, b, c, d = solution

print('a =', a)
print('b =', b)
print('c =', c)
print('d =', d)

print('First Derivative :', simplify(b))
