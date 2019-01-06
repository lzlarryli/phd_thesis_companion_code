import matplotlib.pyplot as plt
import sympy
from fenics import Expression, interpolate
from regge_geodesics import exponential_map
from utils import plot_mesh, exact_kepler
from mshr import Circle, generate_mesh
from sympy2fenics import sympy2exp

# exact kepler orbit
H = -1.5
L = 0.5
(q0, p0, _, S, (c, a, b), _, _) = exact_kepler(H, L)

# symbolic computation for the jacobi metric for schwarzschild geodesics
x, y = sympy.var('x[0], x[1]')
E, m, M = sympy.var('E, m, M')
r = sympy.sqrt(x * x + y * y)
g = E**2 - m**2 + 2 * M * m**2 / r
f = 1 / (1 - 2 * M / r)
ds = f * g / r**2 * sympy.Matrix(((f * x**2 + y**2, (f - 1) * x * y),
                                  ((f - 1) * x * y, x**2 + f * y**2)))
# choose parameters so that the schwarzschild solution is close to the kelper
# the most import parameter is M.
M = 0.0025
m = 1.0 / M**0.5
E = (2.0 * H + m**2)**0.5
gexp = Expression(sympy2exp(ds), E=E, m=m, M=M, degree=8)

# build a large enough domain
domain = Circle(Point(c), b + 0.45)
mesh = generate_mesh(domain, 24)

# Regge 0 is used
g = interpolate(gexp, FunctionSpace(mesh, "Regge", 1))
# ODE solver step size
h = min([c.inradius() for c in cells(mesh)]) / 2.0
# solve
(ts, solh) = exponential_map(g, 0, q0, p0, h, 30 * S)
# plot
(qh, _) = solh(ts)
plt.plot(qh[:, 0], qh[:, 1], color="blue")
plot_mesh(mesh)
plt.show()
