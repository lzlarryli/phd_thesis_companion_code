""" Collection of auxiliary code."""
import numpy as np
from fenics import Mesh, BoundaryMesh

def calc_rate(mesh_sizes, data):
    """ Compute the rate of converge by tabulating the successive slopes."""
    mesh_sizes = np.array(mesh_sizes)
    data = np.array(data)
    tmp = np.diff(np.log(data)) / np.diff(np.log(mesh_sizes))
    rate = np.zeros(data.size)
    rate[0] = np.nan
    rate[1:] = tmp
    return rate

def gaussian_mesh_randomizer(mesh, percentage, preserve_boundary=True):
    """
    Randomly perturb a given mesh.

    Args:
        mesh: Input mesh.
        percentage: Maximum perturbation in percentage of mesh.hmin().
        preserve_boundary: Whether to move the vertices on the boundary.
    Returns:
        rmesh: The perturbed mesh.
    """
    rmesh = Mesh(mesh)
    deltax = (np.random.randn(rmesh.num_vertices(), rmesh.geometry().dim())
              * percentage * rmesh.hmin())
    if preserve_boundary:
        boundary_mesh = BoundaryMesh(rmesh, "exterior")
        boundary_vertices = boundary_mesh.entity_map(0).array()
        deltax[boundary_vertices] = 0.0
    rmesh.coordinates()[:] = rmesh.coordinates() + deltax
    return rmesh
