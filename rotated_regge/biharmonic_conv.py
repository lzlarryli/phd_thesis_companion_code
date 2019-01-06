"""Test the convergence rate of S(REG)xCG for the biharmonic equation."""

import ast
import argparse

from tabulate import tabulate
from fenics import *
from mshr import *

import sympy2fenics as sf
from misc import calc_rate, gaussian_mesh_randomizer

# FEniCS global parameters
set_log_level(ERROR)
parameters["form_compiler"]["representation"] = "uflacs"

def test():
    """Test the convergence rate of S(REG)xCG for the biharmonic equation in
    2D and 3D.

    Args:
        dim (int): Dimension, either 2 or 3.
        degree (int): SREG(degree) x CG(degree + 1) will be used.
        sizes (list): Mesh sizes to test. This should be given as [2,4,8].
    """
    # Parse input
    parser = argparse.ArgumentParser()
    parser.add_argument("dim", type=int)
    parser.add_argument("degree", type=int)
    parser.add_argument("mesh_sizes", type=ast.literal_eval)
    args = parser.parse_args()
    dim = args.dim
    degree = args.degree
    mesh_sizes= args.mesh_sizes

    # Exact solution
    ext_degree = 5  # Degree for the representation of the exact solution
    if dim == 2:
        sol = sf.str2sympy("sin(pi*x)*sin(pi*x)*sin(pi*y)*sin(pi*y)")
    else:
        sol = sf.str2sympy("sin(pi*x)*sin(pi*x)*sin(pi*y)*sin(pi*y)*sin(pi*z)"
                           "*sin(pi*z)")
    u_ext = Expression(sf.sympy2exp(sol), degree=ext_degree)
    f_ext = Expression(sf.sympy2exp(sf.div(sf.grad(sf.div(sf.grad(sol))))),
                       degree=ext_degree)
    sigma = sf.hess(sol)
    sigma_ext = Expression(sf.sympy2exp(sigma), degree=ext_degree)
    gradu_ext = Expression(sf.sympy2exp(sf.grad(sol)), degree=ext_degree)

    # Compute the convergence rates
    hs = []
    eu_L2 = []
    eu_H1 = []
    esigma_L2 = []
    for m in mesh_sizes:
        # Mesh
        if dim == 2:
            domain = Rectangle(Point(0, 0), Point(1, 1))
            mesh = generate_mesh(domain, m)
        else:
            mesh = gaussian_mesh_randomizer(UnitCubeMesh(m, m, m), 0.1)
        hs.append((mesh.hmax() + mesh.hmin()) / 2.0)

        # Problem setup
        REG = FiniteElement('Regge', mesh.ufl_cell(), degree)
        CG = FiniteElement('CG', mesh.ufl_cell(), degree + 1)
        V = FunctionSpace(mesh, REG * CG)
        (gamma, u) = TrialFunctions(V)
        (mu, v) = TestFunctions(V)
        S = lambda mu: mu - Identity(dim) * tr(mu)
        def a(gamma, mu):
            return inner(S(gamma), S(mu)) * dx
        def b(mu, v):
            n = FacetNormal(mesh)
            return inner(S(mu), grad(grad(v))) * dx \
              - dot(dot(S(mu('+')), n('+')), n('+')) * jump(grad(v), n) * dS \
              - dot(dot(S(mu), n), n) * dot(grad(v), n) * ds
        B = a(gamma, mu) - b(mu, u) + b(gamma, v)
        L = f_ext * v * dx
        bc = DirichletBC(V.sub(1), Constant(0.0),
                         lambda x, on_boundary: on_boundary)

        # Solve
        w_h = Function(V)
        print("Solve for m={}...".format(m), end="")
        solve(B == L, w_h, bc, solver_parameters={"linear_solver": "mumps"})
        print("done.")
        (gamma_h, u_h) = w_h.split()

        # Error estimation
        ue = interpolate(u_ext, FunctionSpace(mesh, 'CG', degree + 2))
        err = ue - u_h
        eu_L2.append(sqrt(assemble(err * err * dx)))
        gradue = interpolate(gradu_ext,
                             VectorFunctionSpace(mesh, 'DG', degree + 1))
        err = gradue - grad(u_h)
        eu_H1.append(sqrt(assemble(dot(err, err) * dx)))
        sigmae = interpolate(sigma_ext,
                             TensorFunctionSpace(mesh, 'DG', degree + 1))
        gamma_h = interpolate(gamma_h,
                              TensorFunctionSpace(mesh, 'DG', degree + 1))
        err = sigmae - S(gamma_h)
        esigma_L2.append(sqrt(assemble(inner(err, err) * dx)))

        # Compute convergence rates
        eu_L2r = calc_rate(hs, eu_L2)
        eu_H1r = calc_rate(hs, eu_H1)
        esigma_L2r = calc_rate(hs, esigma_L2)
        # Print table
        headers = ["Mesh size", "‖u‖", "Rate", "‖∇u‖", "Rate", "‖σ‖", "Rate"]
        table = zip(*[mesh_sizes, eu_L2, eu_L2r, eu_H1, eu_H1r, esigma_L2,
                      esigma_L2r])
        #format = "fancy_grid" # Pretty printing for the terminal
        format = "latex_booktabs" # Output LaTeX table
        print(tabulate(table, headers=headers, tablefmt=format,
                       floatfmt=("d",) + ("e", "0.2f") * 3))

if __name__ == "__main__":
    test()
