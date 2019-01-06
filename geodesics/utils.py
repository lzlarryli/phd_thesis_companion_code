import numpy
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from fenics import cells, Expression, Point, RectangleMesh
from mshr import Ellipse, generate_mesh
from scipy.integrate import quad
TOL = 1e-10


def plot_mesh(mesh, color="green", alpha=0.5):
    """ Plot 2D mesh."""
    coors = mesh.coordinates()
    trigs = numpy.asarray([cell.entities(0) for cell in cells(mesh)])
    trimesh = tri.Triangulation(coors[:, 0], coors[:, 1], trigs)
    plt.triplot(trimesh, color=color, alpha=alpha)
    (x0, y0) = numpy.min(mesh.coordinates(), axis=0)
    (x1, y1) = numpy.max(mesh.coordinates(), axis=0)
    plt.axis([x0, x1, y0, y1])
    plt.axes().set_aspect('equal')


def plot_line(p, q, color='black'):
    """ Draw line from p to q. """
    plt.plot([p[0], q[0]], [p[1], q[1]], color)
    plt.show()


def plot_trig(c, color='black'):
    """ Draw a 2D mesh cell. """
    vs = c.get_vertex_coordinates().reshape((3, 2))
    plot_line(vs[0], vs[1], color)
    plot_line(vs[1], vs[2], color)
    plot_line(vs[2], vs[0], color)
    plt.show()


def rate_plot(hs, es, marker='D', name=None):
    """ Convergence rate plot. """
    plt.figure()
    lh = numpy.log(hs)[::-1]
    le = numpy.log(es)[::-1]
    (k, b) = numpy.polyfit(lh, le, 1)
    r = round(k)
    print('The estimated error rate is {} with k = {}.'.format(r, k))
    lr = b + r * lh
    plt.plot(lh, le, color='blue', marker=marker, label='error')
    plt.plot(lh, lr, color='green', label='ref k={}'.format(r))
    plt.legend(loc='best')
    if name:
        plt.savefig(name + '.svg', format='svg')
    else:
        plt.show()


def calc_rate(hs, data):
    """ Compute the rate of converge by tabulating the successive slopes."""
    hs = numpy.array(hs)
    data = numpy.array(data)
    tmp = numpy.diff(numpy.log(data))/numpy.diff(numpy.log(hs))
    rate = numpy.zeros(data.size)
    rate[1:] = tmp
    return rate


def kepler_jacobi_metric(c):
    "The expression for the Jacobi metric of the Kelper system at energy c."
    return Expression(
        (("2.0*(c+1.0/sqrt(x[0]*x[0]+x[1]*x[1]+DOLFIN_EPS))", "0.0"),
         ("0.0", "2.0*(c+1.0/sqrt(x[0]*x[0]+x[1]*x[1]+DOLFIN_EPS))")),
        c=c, degree=10)


def exact_kepler(H, L):
    """The exact solution to the Kelper's problem with energy H and initial
    angular momentum L. At t=0, the planet is on the x+ axis.
    """
    e = numpy.sqrt(1.0 + 2.0 * H * L * L)   # eccentricity
    a = L * L / (1.0 - e * e)               # semi-major axis
    b = a * numpy.sqrt(1.0 - e * e)         # semi-minor axis
    T = 2.0 * numpy.pi * a ** 1.5           # period
    f = e * a                               # focus
    c = numpy.array([-f, 0])                # origin is at the focus
    ellipse = (c, a, b)                     # parameter for the ellipse

    # solve M = E - e sin(E)
    @numpy.vectorize
    def Esol(t):
        M = t / a ** 1.5        # mean anomaly
        E = M                   # initial guess for eccentric anomaly
        while True:             # Newton's method
            Enew = E + (M + e * numpy.sin(E) - E) / (1.0 - e * numpy.cos(E))
            if abs(E - Enew) < TOL:
                return Enew
            E = Enew

    # exact solution
    def sol(t):
        E = Esol(t)
        return numpy.array([a * numpy.cos(E) - f, b * numpy.sin(E)]).T
    # compute initial data
    q0 = numpy.array(sol(0))              # initial position q0=(q0x, 0.0)
    p0y = L / q0[0]
    p0x = numpy.sqrt((H + 1.0 / q0[0]) * 2.0 - p0y ** 2.0)
    p0 = numpy.array([p0x, p0y])

    # compute reparameterization
    def integrand(s):
        return 2.0 * (H + 1.0 / (a * (1.0 - e * numpy.cos(Esol(s)))))
    S = quad(integrand, 0, T, epsrel=TOL, epsabs=TOL)[0]     # period in s

    @numpy.vectorize
    def t2s(t):
        n = t // T
        r = t % T
        return n * S + quad(integrand, 0, r, epsrel=TOL, epsabs=TOL)[0]
    return (q0, p0, T, S, ellipse, sol, t2s)


def integrals(qs, ps):
    """ Compute the energy and anglar momentum given ps and qs. """
    H = [p.dot(p) / 2.0 - 1.0 / numpy.sqrt(q.dot(q)) for (q, p) in zip(qs, ps)]
    L = [q[0] * p[1] - q[1] * p[0] for (q, p) in zip(qs, ps)]
    return (numpy.array(H), numpy.array(L))


def reference_data(H, L, mesh_type, mesh_size, padding=0.07):
    g = kepler_jacobi_metric(c=H)
    (q0, p0, _, _, (c, a, b), _, _) = exact_kepler(H, L)
    if mesh_type == 'uniform':
        # uniform rectangular mesh containing the orbit
        mesh = RectangleMesh(Point(c[0] - a - padding, c[1] - b - padding),
                             Point(c[0] + a + padding, c[1] + b + padding),
                             mesh_size, mesh_size)
    else:
        # unstructured annular mesh containing the orbit
        ell_out = Ellipse(Point(c), a + padding, b + padding)
        ell_in = Ellipse(Point(c), a - padding, b - padding)
        domain = ell_out - ell_in
        mesh = generate_mesh(domain, mesh_size)
    return (q0, p0, g, mesh)
