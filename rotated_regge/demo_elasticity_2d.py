from fenics import *
from mshr import *

# Element: SREG(r) x N2curl(r)
r = 1

# Domain
width = 1.0
length = 3.0
domain = Rectangle(Point(0.0, 0.0), Point(length, width)) \
    - Circle(Point(0.4, 0.3), 0.2) \
    - Circle(Point(1.5, 0.5), 0.375) \
    - Circle(Point(2.4, 0.6), 0.3)
mesh = generate_mesh(domain, 120)

# Finite elements
SREG = FiniteElement('HHJ', mesh.ufl_cell(), r)
NED = FiniteElement('N2curl', mesh.ufl_cell(), r)
V = FunctionSpace(mesh, SREG * NED)
(sigma, u) = TrialFunctions(V)
(tau,   v) = TestFunctions(V)

# Boundary conditions
# Left-side is clamped u=(0,0)
bc_left = DirichletBC(V.sub(1), Constant((0.0, 0.0)),
                      lambda x, on_boundary: on_boundary and near(x[0], 0.0))
# Right-side is compressed u=(-1,0)
g = Constant((-1.0, 0))
#  u.n=g.n is natural boundary condition
class Right(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], length)
boundary_parts = FacetFunction("size_t", mesh)
boundary_parts.set_all(0)
Right().mark(boundary_parts, 1)
n = FacetNormal(mesh)
gn = dot(dot(tau, n), n) * dot(g, n) * ds(subdomain_data=boundary_parts)(1)
#  u.t=g.t is essential boundary
bc_right = DirichletBC(V.sub(1), g, lambda x, on_boundary:
                       on_boundary and near(x[0], length))
# Top and bottom are traction-free
bc_traction1 = DirichletBC(V.sub(0), Constant(((0.0, 0.0), (0.0, 0.0))),
                            lambda x, on_boundary: on_boundary
                            and (near(x[1], 0.0) or near(x[1], width)))
# Holes are traction-free
bc_traction2 = DirichletBC(V.sub(0), Constant(((0.0, 0.0), (0.0, 0.0))),
                            lambda x, on_boundary: on_boundary
                            and (x[0] > DOLFIN_EPS
                                 and x[0] < length - DOLFIN_EPS
                                 and x[1] > DOLFIN_EPS
                                 and x[1] < width - DOLFIN_EPS))

# Left-hand side
E  = 10.0
nu = 0.2
mu = E / (2.0 * (1.0 + nu))
l = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
def A(sigma, dim):
    return 1.0 / (2.0 * mu) \
        * (sigma - l / (2.0 * mu + dim * l) * tr(sigma) * Identity(dim))
def a(sigma, tau):
    return inner(A(sigma, 2), tau) * dx
def b(tau, v):
    n = FacetNormal(mesh)
    return - (inner(tau, sym(grad(v)))) * dx \
        + dot(dot(tau('+'), n('+')), n('+')) * jump(v, n) * dS \
        + dot(dot(tau, n), n) * dot(v, n) * ds
B = a(sigma, tau) + b(tau, u) + b(sigma, v)
# Body force
bf = - dot(Constant((0.0, 0.0)), v) * dx
L = bf + gn

# Solve
w_h = Function(V)
bcs = [bc_left, bc_right, bc_traction1, bc_traction2]
solve(B == L, w_h, bcs, solver_parameters={"linear_solver": "mumps"})
(sigma_h, u_h) = w_h.split()

# Compute von Mises stress
deviator = sigma_h - tr(sigma_h) / 2.0 * Identity(2)
vms = project(inner(deviator, deviator), FunctionSpace(mesh, "DG", r))

File("plots/2d_elasticity_{:.8}.pvd".format(nu)) << u_h
File("plots/2d_elasticity_{:.8}_vms.pvd".format(nu)) << vms
