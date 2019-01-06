import matplotlib.pyplot as plt
from fenics import FunctionSpace, cells, interpolate
from regge_geodesics import exponential_map
from utils import plot_mesh, reference_data

# get initial data and mesh for a Kelper problem with
# energy H=-1.5 (elliptic orbit) and initial momentum L=0.5.
(q0, p0, gexp, mesh) = reference_data(H=-1.5, L=0.5, mesh_type='uniform',
                                      mesh_size=24, padding=0.07)
# Regge 0 is used
g = interpolate(gexp, FunctionSpace(mesh, "Regge", 0))
# ODE solver step size
h = min([c.inradius() for c in cells(mesh)]) / 2.0
# solve
(ts, solh) = exponential_map(g, 0, q0, p0, h, 10.0)
# plot
(qh, _) = solh(ts)
plt.plot(qh[:, 0], qh[:, 1], color="blue")
plot_mesh(mesh)
plt.show()
