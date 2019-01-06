""" Long time error behavior for Regge geodesic solver.
    python python exp_long_time_behavior.py [degree] [mesh_size]
"""
import matplotlib.pyplot as plt
from fenics import FunctionSpace, cells, interpolate
from regge_geodesics import exponential_map
from utils import (reference_data, exact_kepler, integrals, rate_plot)
import numpy
import sys

# degree
degree = int(sys.argv[1])
# mesh size
n = int(sys.argv[2])
# how many orbits to run before assessing the error
M = int(sys.argv[3])

# prefix for the output names
prefix = 'extra_long_time'
# number of points per orbit to sample for the error computation
SAMPLES = 8 # 40
# mesh padding
padding = 0.07
# kepler parameters
H = -1.5
L = 0.5

# initialize
(q0, p0, T, S, (c, a, b), sol, t2s) = exact_kepler(H, L)
Tmax = T * M
(_, _, gexp, mesh) = reference_data(H, L, 'unstructured', n, padding=padding)
g = interpolate(gexp, FunctionSpace(mesh, "Regge", degree))
h = min([c.inradius() for c in cells(mesh)]) / 2.0

# solve
(_, solh) = exponential_map(g, 0, q0, p0, h, t2s(Tmax), verbose=True)

# file name format
prefix = prefix + '-deg{}'.format(degree)
print('')
print('==== Degree {} ===='.format(degree))
# sample time
t = numpy.linspace(0, Tmax, SAMPLES * M + 1)
# a fast way to compute s = t2s(t)
tt = numpy.linspace(0, T, SAMPLES + 1)
ss = t2s(tt)
s = []
for i in range(M):
    s.append(ss[0:SAMPLES] + i * numpy.repeat(S, SAMPLES))
s = numpy.append(numpy.concatenate(s), M * S)

# evaluate exact and discrete solutions
qe = sol(t)
(qh, ph) = solh(s)

# compute error
d = qh - qe
e = numpy.array([numpy.sqrt(q.dot(q)) for q in d])
plt.figure()
plt.plot(s, e, color='blue')
plt.savefig(prefix + 'error.svg', format='svg')

# plot error rate
rate_plot(s[1:], e[1:], marker='', name=prefix + 'e')

# relative error in energy and momentum
(Hh, Lh) = integrals(qh, ph)

eE = (Hh - H) / H
plt.figure()
plt.plot(s, eE)
plt.savefig(prefix + 'H.svg', format='svg')

eL = (Lh - L) / L
plt.figure()
plt.plot(s, eL)
plt.savefig(prefix + 'L.svg', format='svg')
print('')
