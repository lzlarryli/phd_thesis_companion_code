"""Solving the standard 1D wave equation using the spacetime CG-CG method.

Equation:

    u'' - uxx = f, in [0,T]x[0,1],
    u(0,x) = u0(x), u'(0,x) = u1(x), for x in [0,1],
    u(t,0) = u(t,1) = 0, for t in [0,T].

Mesh:

    Space-time mesh for [0,T]x[0,1]

Weak formulation: find u in H1([0,T], H01[0,1]) with the initial condition,

    - (u', v') + (ux, vx) = (f, v) + [u1, v(0)]

for all v in H1([0,T], H01[0,1]) satisfying the terminal boundary condition

    v(T,x)=0.
"""

from fenics import (Mesh, BoundaryMesh, SubDomain, RectangleMesh, Point,
                    FunctionSpace, MeshFunction, TestFunction, TrialFunction,
                    Function, LUSolver, DirichletBC, PETScVector, PETScMatrix,
                    Constant, MeshEditor,
                    near, as_backend_type, ds, dot, grad, solve, assemble, dx)
import numpy as np
from petsc4py import PETSc


def mesh_randomizer_2d(mesh, percentage, preserve_boundary=True):
    """
    Randomly perturb a given mesh.

    Args:
        mesh: Input mesh.
        percentage: Maximum perturbation in percentage of mesh.hmin().
        preserve_boundary: Whether to move the vertices on the boundary.
    Returns:
        rmesh: The perturbed mesh.
    """
    # Generate a deep copy of the mesh
    rmesh = Mesh(mesh)
    meshsize = rmesh.hmin()
    # Randomly perturbed the mesh
    radius = np.random.rand(rmesh.num_vertices()) * percentage * meshsize
    theta = np.random.rand(rmesh.num_vertices()) * 2.0 * np.pi
    deltax = np.zeros([rmesh.num_vertices(), 2])
    deltax[:, 0] = (radius * np.sin(theta)).transpose()
    deltax[:, 1] = (radius * np.cos(theta)).transpose()
    # What to do with the boundary vertices
    if preserve_boundary:
        # Exterior means global boundary
        boundary_mesh = BoundaryMesh(rmesh, "exterior")
        # entity_map contains the indices of vertices on the boundary
        boundary_vertices = boundary_mesh.entity_map(0).array()
        deltax[boundary_vertices] = 0.0
    rmesh.coordinates()[:] = rmesh.coordinates() + deltax
    return rmesh


def get_dof_by_criterion(space, criterion):
    """ Return dofs with coordinates satisfying a given condition.

    Args:
        space: The function space.
        criterion: A boolean function which takes a coordinate (numpy array).

    Outputs:
        dof_no: A list of global indices of dofs where criterion is True.
        dof_coor: A list corresponds to the coordinates of dofs in dof_no.
    """
    gdim = space.mesh().geometry().dim()
    dof_coors = space.tabulate_dof_coordinates().reshape((-1, gdim))
    return list(zip(*[(i, coor)
                      for (i, coor) in enumerate(dof_coors)
                      if criterion(coor)]))


class TemporalSlice(SubDomain):
    """A temporal slice of the space-time domain."""
    def __init__(self, time):
        self.time = time
        SubDomain.__init__(self)

    def inside(self, coor, on_boundary):
        return near(coor[0], self.time)


class Boundary(SubDomain):
    """Spatial boundary of the space-time domain."""
    def __init__(self, left, right):
        self.left = left
        self.right = right
        SubDomain.__init__(self)

    def inside(self, pos, on_boundary):
        return (near(pos[1], self.left)) or (near(pos[1], self.right))


class SpaceTimeDomain:
    """(1+1) space-time domain [t0, t1] x [x0, x1]."""
    def __init__(self, t0, t1, x0, x1):
        self.t0 = t0
        self.t1 = t1
        self.x0 = x0
        self.x1 = x1

    def get_initial_slice(self):
        """Generate temporal domains for marking mesh."""
        return TemporalSlice(self.t0)

    def get_terminal_slice(self):
        """Generate temporal domains for marking mesh."""
        return TemporalSlice(self.t1)

    def get_spatial_boundary(self):
        """Generate spatial domains for marking mesh."""
        return Boundary(self.x0, self.x1)

    def get_uniform_mesh(self, temporal_nodes, spatial_nodes):
        """Generate uniform mesh of the spacetime."""
        return RectangleMesh(Point(self.t0, self.x0),
                             Point(self.t1, self.x1),
                             temporal_nodes, spatial_nodes)

def apply_time_boundary_conditions(domain, V, u0, A, b):
    """Apply the time slice boundary conditions by hand.

    Args:
        domain: Space-time domain.
        V: Function space.
        u0: Initial data.
        A: The stiffness matrix.
        b: The right-hand side.
    Outputs:
        A: The new stiffness matrix with the boundary conditions.
        b: The new right-hand side with the boundary conditions.
    """
    # Export matrices to PETSc
    A = as_backend_type(A).mat()
    b = as_backend_type(b).vec()
    # Apply terminal boundary condition on v by zeroing the corresponding
    # matrix rows. The dof indices are saved for later.
    def on_terminal_slice(x):
        return domain.get_terminal_slice().inside(x, True) \
            and (not domain.get_spatial_boundary().inside(x, True))
    (rows_to_zero, _) = get_dof_by_criterion(V, on_terminal_slice)
    A.zeroRows(rows_to_zero, diag=0)
    # Apply initial boundary condition on u
    def on_initial_slice(x):
        return domain.get_initial_slice().inside(x, True) \
            and (not domain.get_spatial_boundary().inside(x, True))
    (dof_no, dof_coor) = get_dof_by_criterion(V, on_initial_slice)
    # Update the matrices
    A.setOption(PETSc.Mat.Option.NEW_NONZERO_LOCATION_ERR, 0)
    for (i, k) in enumerate(dof_no):
        j = rows_to_zero[i]
        A[j, k] = 1.0
        b[j] = u0(dof_coor[i])
    A.assemble()
    b.assemble()

    # put petsc4py matrix back to fenics
    A = PETScMatrix(A)
    b = PETScVector(b)
    return (A, b)

def solve_wave_equation(u0, u1, u_boundary, f, domain, mesh, degree):
    """Solving the wave equation using CG-CG method.

    Args:
        u0: Initial data.
        u1: Initial velocity.
        u_boundary: Dirichlet boundary condition.
        f: Right-hand side.
        domain: Space-time domain.
        mesh: Computational mesh.
        degree: CG(degree) will be used as the finite element.
    Outputs:
        uh: Numerical solution.
    """
    # Element
    V = FunctionSpace(mesh, "CG", degree)
    # Measures on the initial and terminal slice
    mask = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
    domain.get_initial_slice().mark(mask, 1)
    ends = ds(subdomain_data=mask)
    # Form
    g = Constant(((-1.0, 0.0), (0.0, 1.0)))
    u = TrialFunction(V)
    v = TestFunction(V)
    a = dot(grad(v), dot(g, grad(u))) * dx
    L = f * v * dx + u1 * v * ends(1)
    # Assembled matrices
    A = assemble(a, keep_diagonal=True)
    b = assemble(L, keep_diagonal=True)
    # Spatial boundary condition
    bc = DirichletBC(V, u_boundary, domain.get_spatial_boundary())
    bc.apply(A, b)
    # Temporal boundary conditions (by hand)
    (A, b) = apply_time_boundary_conditions(domain, V, u0, A, b)
    # Solve
    solver = LUSolver()
    solver.set_operator(A)
    uh = Function(V)
    solver.solve(uh.vector(), b)
    return uh


def unit_mesh(ht, hx):
    editor = MeshEditor()
    mesh = Mesh()
    editor.open(mesh, "triangle", 2, 2)
    editor.init_vertices(7)
    editor.add_vertex(0, np.array([0.0, 0.0]))
    editor.add_vertex(1, np.array([ht / 2.0, 0.0]))
    editor.add_vertex(2, np.array([0.0, hx / 2.0]))
    editor.add_vertex(3, np.array([ht / 2.0, hx / 2.0]))
    editor.add_vertex(4, np.array([ht, hx / 2.0]))
    editor.add_vertex(5, np.array([ht / 2.0, hx]))
    editor.add_vertex(6, np.array([ht, hx]))
    editor.init_cells(6)
    editor.add_cell(0, np.array([0, 1, 3], dtype=np.uintp))
    editor.add_cell(1, np.array([0, 2, 3], dtype=np.uintp))
    editor.add_cell(2, np.array([1, 3, 4], dtype=np.uintp))
    editor.add_cell(3, np.array([2, 3, 5], dtype=np.uintp))
    editor.add_cell(4, np.array([3, 4, 6], dtype=np.uintp))
    editor.add_cell(5, np.array([3, 5, 6], dtype=np.uintp))
    editor.close()
    mesh.order()
    return mesh
