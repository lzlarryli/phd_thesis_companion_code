from fenics import *
from mshr import *

# Element SREG(r) x N2curl(r)
r = 1

# Domain
d0 = 4.0
d1 = 2.0
d2 = 1.0
domain = Box(Point(0.0, 0.0, 0.0), Point(d0, d1, d2)) \
         - Cylinder(Point(d0 / 3.0, 0, d2 / 2.0),
                    Point(d0 / 3.0, d1, d2 / 2.0), 0.3, 0.3) \
         - Cylinder(Point(d0 * 2.0 / 3.0, d1 / 2.0, 0),
                    Point(d0 * 2.0 / 3.0, d1 / 2.0, d2), 0.7, 0.7)
mesh = generate_mesh(domain, 42)

# Finite elements
REG = FiniteElement('Regge', mesh.ufl_cell(), r)
NED = FiniteElement('N2curl', mesh.ufl_cell(), r)
V   = FunctionSpace(mesh, REG * NED)
(gamma, u) = TrialFunctions(V)
(tau,   v) = TestFunctions(V)

# Material coefficients: the a-form
E  = 1.0
nu = 0.2
mu = E / (2.0 * (1.0 + nu))
l = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
# A(S(gamma), S(tau)) expanded by hand
def a(gamma, tau):
    k = l / (2.0 * mu + 3.0 * l)
    return 1.0 / (2.0 * mu) * (inner(gamma, tau)
                               + (1.0 - 4.0 * k) * tr(gamma) * tr(tau)) * dx

# Distributional div pairing: the b-form
S = lambda tau: tau - Identity(3) * tr(tau)
def b(tau, v):
    n = FacetNormal(mesh)
    return - (inner(S(tau), sym(grad(v)))) * dx \
        + dot(dot(S(tau('+')), n('+')), n('+')) * jump(v, n) * dS \
        + dot(dot(S(tau), n), n) * dot(v, n) * ds

B = a(gamma, tau) + b(tau, u) + b(gamma, v)

# Left-side is clamped u=(0,0,0)
bc_left = DirichletBC(V.sub(1), Constant((0.0, 0.0, 0.0)),
                      lambda x, on_boundary: on_boundary and near(x[0], 0.0))

# Right-side is rotated
class right_end_displacement(Expression):
    def eval(self, values, x):
      phi = pi / 6.0
      z1 = x[1] - 0.5 * d1
      z2 = x[2] - 0.5 * d2
      values[0] = 0.0
      values[1] = cos(phi) * z1 - sin(phi) * z2 - z1
      values[2] = sin(phi) * z1 + cos(phi) * z2 - z2
    def value_shape(self):
        return (3, )
g = right_end_displacement(degree=5)
# u.n=g.n is natural boundary condition
class Right(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], d0)
boundary_parts = FacetFunction("size_t", mesh)
boundary_parts.set_all(0)
Right().mark(boundary_parts, 1)
n = FacetNormal(mesh)
gn = dot(dot(S(tau), n), n) * dot(g, n) * ds(subdomain_data=boundary_parts)(1)
# u.t=g.t is essential boundary
bc_right = DirichletBC(V.sub(1), g, lambda x, on_boundary:
                       on_boundary and near(x[0], d0))

# All other parts are traction-free
zero_stress = Constant(((0.0, 0.0, 0.0),
                        (0.0, 0.0, 0.0),
                        (0.0, 0.0, 0.0)))
bc_traction_side = DirichletBC(V.sub(0), zero_stress,
                               lambda x, on_boundary: on_boundary and
                               (near(x[1], 0) or near(x[1], d1)
                                or near(x[2], 0) or near(x[2], d2)))
bc_traction_inside = DirichletBC(V.sub(0), zero_stress,
                                 lambda x, on_boundary: on_boundary and
                                 (x[0] > DOLFIN_EPS
                                  and x[0] < d0 - DOLFIN_EPS))


# Body force
bf = - dot(Constant((0.0, 0.0, 0.0)), v) * dx
L = bf + gn

# solve
w_h = Function(V)
bcs = [bc_left, bc_right, bc_traction_side, bc_traction_inside]
solve(B == L, w_h, bcs, solver_parameters={"linear_solver": "mumps"})
(gamma_h, u_h) = w_h.split()

# Compute von Mises stress
sigma_h = S(gamma_h)
deviator = sigma_h - tr(sigma_h) / 3.0 * Identity(3)
vms = project(inner(deviator, deviator), FunctionSpace(mesh, "DG", r))

File("plots/3d_elasticity_poisson_ratio_{:.8}.pvd".format(nu)) << u_h
File("plots/3d_elasticity_poisson_ratio_{:.8}_vms.pvd".format(nu)) << vms
