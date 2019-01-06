""" Problem:

   //////////////////////////
   /+-----------------------+
   /|                       |
   /|        xx             |
   /|        xx    |\       |
   /|              |/\      |
   /|              |//\     |
   /+--------------+/ /+----+
   /////////////////  ///////

   Right-end is free.
   All other 3 edges and the cut is clamped.
   The load is in the middle at the 'x'.
"""
from __future__ import print_function
from fenics import *
from mshr import *
import numpy as np
parameters["form_compiler"]["representation"] = "uflacs"

###
# Element: SREG(r) x CG(r+1) #
                           ###
r = 2

# Domain
length = 3.0
width = 2.0
domain = (Rectangle(Point(0.0, 0.0), Point(length, width))
          - Polygon([Point(2.0, 0.8), Point(2.0, 0.0), Point(2.5, 0.0)]))
mesh = generate_mesh(domain, 48)

# Finite element spaces
REG = FiniteElement('Regge', mesh.ufl_cell(), r)
CG = FiniteElement('CG', mesh.ufl_cell(), r + 1)
V = FunctionSpace(mesh, REG * CG)

# Boundary conditions
# The right-end is free
bc_right = DirichletBC(V.sub(0), Constant(((0.0, 0.0), (0.0, 0.0))),
                       lambda x, on_boundary:
                       on_boundary and near(x[0], length))
# The cut is clamped
bc_cut = DirichletBC(V.sub(1), Constant(0.0), lambda x, on_boundary:
                     on_boundary and (x[0] < length - DOLFIN_EPS and
                                      x[1] > DOLFIN_EPS and x[1] < width))
# The other three ends are clamped
bc_clamped = DirichletBC(V.sub(1), Constant(0.0), lambda x, on_boundary:
                         on_boundary and (near(x[0], 0.0) or near(x[1], 0.0)
                                          or near(x[1], width)))

# Right-hand side: load in the middle
class Load(Expression):
    def eval(self, values, x):
        if sqrt((x[0] - length / 2.0)**2 + (x[1] - width / 2.0)**2) < 0.2:
            values[0] = 1.0
        else:
            values[0] = 0.0
    def value_shape(self):
        return ()
f = Load(degree=4)

# Variational formulation
(sigma, u) = TrialFunctions(V)
(tau, v) = TestFunctions(V)
S = lambda mu: mu - Identity(2) * tr(mu)
def a(sigma, tau):
    return inner(S(sigma), S(tau)) * dx
def b(tau, v):
    n = FacetNormal(mesh)
    return inner(S(tau), grad(grad(v))) * dx \
      - dot(dot(S(tau('+')), n('+')), n('+')) * jump(grad(v), n) * dS \
      - dot(dot(S(tau), n), n) * dot(grad(v), n) * ds
B = a(sigma, tau) + b(tau, u) + b(sigma, v)
L = - f * v * dx

# Solve
w_h = Function(V)
bcs = [bc_right, bc_cut, bc_clamped]
solve(B == L, w_h, bcs, solver_parameters={"linear_solver": "mumps"})
(sigma_h, u_h) = w_h.split()
W = FunctionSpace(mesh, CG)
u_sreg = interpolate(u_h, W)



###
# Element: HHJ(r) x CG(r+1) #
                          ###
r = 2

# Finite element spaces
HHJ = FiniteElement('HHJ', mesh.ufl_cell(), r)
V = FunctionSpace(mesh, HHJ * CG)

# Variational formulation
(sigma, u) = TrialFunctions(V)
(tau, v) = TestFunctions(V)
def a(sigma, tau):
    return inner(sigma, tau) * dx
def b(tau, v):
    n = FacetNormal(mesh)
    return inner(tau, grad(grad(v))) * dx \
      - dot(dot(tau('+'), n('+')), n('+')) * jump(grad(v), n) * dS \
      - dot(dot(tau, n), n) * dot(grad(v), n) * ds
B = a(sigma, tau) + b(tau, u) + b(sigma, v)
L = - f * v * dx

# Boundary conditions
# The right-end is free
bc_right = DirichletBC(V.sub(0), Constant(((0.0, 0.0), (0.0, 0.0))),
                       lambda x, on_boundary:
                       on_boundary and near(x[0], length))
# The cut is clamped
bc_cut = DirichletBC(V.sub(1), Constant(0.0), lambda x, on_boundary:
                     on_boundary and (x[0] < length - DOLFIN_EPS and
                                      x[1] > DOLFIN_EPS and x[1] < width))
# The other three ends are clamped
bc_clamped = DirichletBC(V.sub(1), Constant(0.0), lambda x, on_boundary:
                         on_boundary and (near(x[0], 0.0) or near(x[1], 0.0)
                                          or near(x[1], width)))

# Solve
w_h = Function(V)
bcs = [bc_right, bc_cut, bc_clamped]
solve(B == L, w_h, bcs, solver_parameters={"linear_solver": "mumps"})
(sigma_h, u_h) = w_h.split()
u_hhj = interpolate(u_h, W)

# Compare
diff = Function(W)
diff.vector()[:] = u_sreg.vector() - u_hhj.vector()
print("The L2-norm difference between the displacement variable is {}."
      .format(np.sqrt(assemble(diff * diff * dx))))
