from fenics import FunctionSpace, interpolate, File
from utils import reference_data, exact_kepler

# get initial data and mesh for a Kelper problem with
# energy H=-1.5 (elliptic orbit) and initial momentum L=0.5.
H = -1.5
L = 0.5
(_, _, gexp, mesh) = reference_data(H=H, L=L, mesh_type='unstructured',
                                    mesh_size=24, padding=0.07)
(q0, p0, T, S, (c, a, b), sol, t2s) = exact_kepler(H, L)

g = interpolate(gexp, FunctionSpace(mesh, "Regge", 1))

File("plots/kepler_metric.pvd") << g
