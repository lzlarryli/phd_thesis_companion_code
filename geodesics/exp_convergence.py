""" Convergence test for Regge geodesic solver.
    python exp_convergence.py [degree] [im] [nm]
    This runs the geodesic solver on meshes of sizes (im, im*2, ..., im*(nm-1))
"""
from fenics import FunctionSpace, cells, interpolate
from regge_geodesics import exponential_map
from utils import (reference_data, exact_kepler, integrals,
                   kepler_jacobi_metric, rate_plot, calc_rate)
import numpy
import sys

# how many orbits to run before assess error
M = 1.65
# mesh padding
padding = 0.07
# kepler parameters
H = -1.5
L = 0.5

# initialize
degree = int(sys.argv[1])
levels = [int(sys.argv[2]) * 2 ** i for i in range(int(sys.argv[3]))]
(q0, p0, T, S, (c, a, b), sol, t2s) = exact_kepler(H, L)
Tmax = T * M
gexp = kepler_jacobi_metric(c=H)
ee = []
eH = []
eL = []
hs = []

# main loop
for n in levels:
    (_, _, _, mesh) = reference_data(H, L, 'unstructured', n, padding=padding)
    hs.append(numpy.mean([c.h() for c in cells(mesh)]))

    print('Compute the solution for n={}...'.format(n))
    g = interpolate(gexp, FunctionSpace(mesh, 'Regge', degree))
    h = min([c.inradius() for c in cells(mesh)]) / 2.0

    (_, solh) = exponential_map(g, 0, q0, p0, h, t2s(Tmax), verbose=True)

    print('Evaluate the solution and compute the error...')
    t = numpy.linspace(0, Tmax, 200 * M + 1)
    s = t2s(t)
    qe = sol(t)
    (qh, ph) = solh(s)
    (Hh, Lh) = integrals(qh, ph)
    d = qe - qh
    ee.append(numpy.max(numpy.sqrt(numpy.array([q.dot(q) for q in d]))))
    eH.append(numpy.max(numpy.abs(Hh - H)))
    eL.append(numpy.max(numpy.abs(Lh - L)))

# compute error rates and output
prefix = 'conv-deg{}-'.format(degree)
print('')
print('===DEGREE {}==='.format(degree))
rate_plot(hs, ee, name=prefix + 'e')
print(calc_rate(hs, ee))
rate_plot(hs, eH, name=prefix + 'H')
print(calc_rate(hs, eH))
rate_plot(hs, eL, name=prefix + 'L')
print(calc_rate(hs, eL))
print('')
