"""
   Geodesic solver for generalized Regge metrics.
"""

from __future__ import print_function
from fenics import (parameters, DOLFIN_SQRT_EPS, Point, Cell, Constant,
                    project, TensorFunctionSpace, facets, DOLFIN_EPS_LARGE)
from scipy.optimize import fixed_point
from scipy.interpolate import BarycentricInterpolator
from numpy import array, sqrt, zeros, concatenate, split, isscalar, nan
from numpy.linalg import inv
from numpy.random import rand
parameters["form_compiler"]["representation"] = "uflacs"

# +--------------------------------------+
# | Geodesic solver on generalized Regge |
# +--------------------------------------+


def exponential_map(g, t0, q0, p0, h, T, verbose=True):
    """ Compute the generalized exponential map on a Regge metric

    Input
         g     a Regge metric
         t0    starting time
         q0    initial position
         p0    initial momentum
         h     ODE solver step size
         T     stopping time

    Output
         ts    time steps
         qh    position as a function of t
         ph    momentum as a function of t
    """
    # initialization
    dim = g.ufl_shape[0]
    mesh = g.function_space().mesh()
    step = _incell_geodesic_step_maker(g)  # ODE solver
    c = _find_cell(g, q0, p0)  # find the starting cell

    # info
    print('Geodesic computation...')
    print(' Initial position: {}'.format(q0))
    print(' Initial momentum: {}'.format(p0))
    print(' Step size: {}. Maximum time: {}'.format(h, T))

    # main loop
    t = t0
    y = concatenate([q0, p0])
    ts = [t0]
    ys = [lambda t: concatenate([q0, p0])]
    while True:
        if verbose:
            print(' progress: {:0.2f}%.'.format(abs(t / T * 100.0)), end='\r')

        # Step 1: solve smooth geodesic equation inside a cell
        while True:
            yn = step(c, t, y, h)
            qn = yn(t + h)[0:dim]
            if c.distance(Point(qn)) > DOLFIN_SQRT_EPS:
                # going out of the cell, break
                break
            # update
            t = t + h
            ts.append(t)
            y = yn
            ys.append(yn)

        # Step 2: truncate the last step to hit the boundary facet
        def dist(k):
            """ distance from the position at (t+k) to the boundary of c"""
            return c.distance(Point(yn(t + k)[0:dim]))
        # bisect to find k such that the position at (t+k)
        # (1) is within DOLFIN_SQRT_EPS of the boundary
        # (2) is outside of the current cell
        a = 0
        k = h
        min_dist = dist(k)
        while min_dist > DOLFIN_SQRT_EPS or abs(a - k) < DOLFIN_EPS_LARGE:
            x = (a + k) / 2.0
            fx = dist(x)
            if fx > 0:
                k = x
                min_dist = fx
            else:
                a = x
        # update (y is updated in the next rotation step)
        t = t + k
        ts.append(t)
        ys.append(yn)

        # Step 3: stop if maximum time T is reached
        if ts[-1] > T:
            break

        # Step 4: rotate p and cross to the next cell if it exists
        # find the facets closest to the end point
        [qn, pn] = split(yn(t), 2)
        f = min(facets(c), key=lambda f: f.distance(Point(qn)))
        # find adjacent cells to the chosen facet
        ncs = f.entities(c.dim())
        if len(ncs) == 1:       # hit domain boundary
            print('Boundary reached at t={}. Terminate.'.format(t))
            break
        else:                   # go to the opposite cell
            # find the next cell
            nc = Cell(mesh, list(set(ncs) - {c.index()})[0])
            # rotate p
            pn = _facet_crossing(pn, g, qn, f, c, nc)
            # update
            y = concatenate([qn, pn])
            c = nc

        # Step 5: check if the new starting point is in c.
        # This failure can happen when the curve goes near a face of low
        # dimension. The small over-crossing (k computed is always greater than
        # the exact value) might end up in another cell. In this case, find the
        # correct cell and restart. This commits an error proportion to
        # DOLFIN_SQRT_EPS in the position without changing the momentum.
        if c.distance(Point(qn)) > 0:
            print(' Warning: the geodesic came near a face of low dimension '
                  'at time {}.'.format(t))
            c = _find_cell(g, qn, pn)

    def solh(t):
        """ The final global piecewise smooth interpolant. """
        # deal with the case when t is just a single number
        if isscalar(t):
            t = array([t])
        # assume t is sorted

        tmp = zeros((len(t), 2 * dim))
        i = 0
        for j in range(len(t)):
            # find the smallest i such that ts[i]>t[j]
            while (i < len(ys)) and (ts[i] < t[j]):
                i = i + 1
            # t[j] should be evaluted using yn[i]
            if i < len(ys):
                tmp[j] = ys[i](t[j])
            else:
                tmp[j] = nan
        return (tmp[:, 0:dim], tmp[:, dim:])    # (q, p) split

    return (array(ts), solh)


def _find_cell(g, q, p):
    """Find the cell q should be in.

    If q is in the interior of a cell c, then return c. If q is in the
    intersection of several cells, return the one that is in the direction of
    q.
    """
    mesh = g.function_space().mesh()
    csi = mesh.bounding_box_tree().compute_entity_collisions(Point(q))
    cs = [Cell(mesh, i) for i in csi]
    cs = [c for c in cs if c.distance(Point(q)) == 0]
    if len(cs) == 0:
        raise RuntimeError('Point {} is not inside any mesh cell. This should '
                           'not be possible.'.format(q))
    elif len(cs) == 1:
        # all good
        return cs[0]
    else:
        # close to faces of lower dimension leading to multiple collisions
        # choose the best cell using p
        def try_step(c):
            ginv = inv(_eval_metric(g, c, q))
            dq = ginv.dot(p)
            qn = q + dq * DOLFIN_SQRT_EPS * 10
            return c.distance(Point(qn))
        return min(cs, key=try_step)


def _eval_metric(g, c, x):
    """ Evaluate the metric g at point x in cell c. """
    gx = zeros(g.value_size())
    g.eval_cell(gx, x, c)
    return gx.reshape(g.ufl_shape)


def _compute_facet_normal(c, f, g, x):
    """ Compute the outward normal to facet f of the cell c under the metric g
    at point x."""
    # get facet tangents
    coors = f.mesh().coordinates()
    vcoors = array([coors[i] for i in f.entities(0)])
    ts = vcoors[1:] - vcoors[0]
    # compute a unit normal
    gx = _eval_metric(g, c, x)
    n = rand(c.dim())
    for v in ts:
        n = n - gx.dot(n).dot(v) / gx.dot(v).dot(v) * v
    n = n / sqrt(gx.dot(n).dot(n))
    # make sure it is outward pointing
    mc = array([c.midpoint()[i] for i in range(c.dim())])
    mf = array([f.midpoint()[i] for i in range(c.dim())])
    if gx.dot(n).dot(mf - mc) < 0:
        n = -n
    return n


def _facet_crossing(p, g, q, f, c, nc):
    """ Computing the facet jump from cell c to cell nc in the momentum at q.

    This is done according to the calculations in the thesis. See Chapter 3.
    """
    n1 = _compute_facet_normal(c, f, g, q)
    n2 = _compute_facet_normal(nc, f, g, q)
    g1 = _eval_metric(g, c, q)
    g2 = _eval_metric(g, nc, q)
    v = inv(g1).dot(p)
    vn = v - g1.dot(n1).dot(v) * (n1 + n2)
    pn = g2.dot(vn)
    return pn


def _incell_geodesic_step_maker(g):
    """ Symplectic integrator of the Hamiltonian geodesic equation in a cell.
    """
    # initialize
    mesh = g.function_space().mesh()
    deg = g.ufl_element().degree()
    dim = g.ufl_shape[0]

    # choose solver and compute metric derivatives
    if deg == 0:
        # lowest degree Regge
        solver = _explicit_euler
        dg = [Constant(((0,) * dim, ) * dim) for i in range(dim)]
    else:
        # for degree >= 1
        solver = _gauss6
        # compute the first derivatives of the metric globally
        # note: this is in fact more efficient than computing the derivative
        # cell by cell on the go. memeory is not a issue here.
        W = TensorFunctionSpace(mesh, 'DG', deg - 1)
        dg = [project(g.dx(i), W, solver_type='mumps') for i in range(dim)]

    def F(y, c):
        """ Hamiltonian geodesic equation as a system."""
        q = y[0:dim]
        p = y[dim:]
        ginv = inv(_eval_metric(g, c, q))
        nq = ginv.dot(p)
        np = array([0.5 * _eval_metric(dg[i], c, q)
                    .dot(ginv.dot(p))
                    .dot(ginv.dot(p))
                    for i in range(dim)])
        return concatenate([nq, np])

    def step(c, t, y0, h):
        """ Geodesic solver wrapper."""
        return solver(lambda y: F(y, c), t, y0, h)
    return step


# +----------------------+
# | Internal ODE solvers |
# +----------------------+

def _explicit_euler(f, t, y, h):
    """ One step of explicit Euler for the autonomous system y'=f(y).

    Input
        f   the right-hand side
        t   the initial time
        y   can either be a number for the initial value or a function whose
            value at t serves as the inital value
        h   step size
    Output
        yn  the solution function on [t, t + h]
    """
    if hasattr(y, '__call__'):
        # y is a function
        y0 = y(t)
    else:
        # y is a single number
        y0 = y
    return BarycentricInterpolator([t, t + h], [y0, y0 + h * f(y0)])


def _gauss6(f, t, y, h):
    """ One step Gauss6 for the autonomous system y'=f(y).

    This is implemented as an Runge-Kutta method.
    See Hairer, Lubich, Wanner, Geometric Numerical Integration, page 30.

    Input
        f   the right-hand side
        t   the initial time
        y   can either be a number for the initial value or a function whose
            value at t serves as the inital value and can be used for
            extrapolating initial guesses for the nonliear solver
        h   step size
    Output
        yn  the solution function on [t, t + h]
    """
    # Step 0: initialize
    (a, b, c) = _WEIGHTS        # get coefficients
    # Step 1: generate initial guess for stage values
    if hasattr(y, '__call__'):
        # y is a function => rename it to l and take y0=y(t)
        l = y
        y0 = y(t)
    else:
        # y is a single number => use the linear interpolant for initial guess
        l = BarycentricInterpolator([t, t + h], [y, y + h * f(y)])
        y0 = y
    # compute initial guess for stage values
    z = array([l(t + ci * h) - y0 for ci in c])

    # Step 2: solve for stage values
    # the nonlinear system associate with the stage values
    def eq(w):
        return h * a.dot(array([f(y0 + wi) for wi in w]))
    # solve using fixed point interation
    z = fixed_point(eq, z, xtol=DOLFIN_SQRT_EPS, maxiter=500000)

    # Step 3: construct the solution as the interpolant
    pts = array([t + ci * h for ci in c])
    return BarycentricInterpolator(pts, y0 + z)

# RK weights for Gauss6
_WEIGHTS = (array([[5.0 / 36.0, 2.0 / 9.0 - sqrt(15.0) / 15.0,
                    5.0 / 36.0 - sqrt(15.0) / 30.0],
                   [5.0 / 36.0 + sqrt(15.0) / 24.0,
                    2.0 / 9.0, 5.0 / 36.0 - sqrt(15.0) / 24.0],
                   [5.0 / 36.0 + sqrt(15.0) / 30.0,
                    2.0 / 9.0 + sqrt(15.0) / 15.0, 5.0 / 36.0]]),
            array([5.0 / 18.0, 4.0 / 9.0, 5.0 / 18.0]),
            array([0.5 - sqrt(15.0) / 10.0, 0.5, 0.5 + sqrt(15.0) / 10.0]))
