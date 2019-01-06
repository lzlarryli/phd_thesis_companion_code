"""
Scalar, vector, and matrix symbolic calculus using sympy with an interface
to generate FEniCS expressions.

By Lizao Li <lzlarryli@gmail.com>

"""


from sympy import symbols, printing, sympify, Matrix, eye


def str2sympy(str):
    """
    Create sympy scalar-, vector-, or matrix-expression from a string.
    Variables (x,y,z) are reserved and used for automatic dimension
    inference. For example:
        f = str2sympy('sin(x)')                   # a scalar function in 1D
        u = str2sympy('(sin(x), sin(y))')         # a vector function in 2D
        w = str2sympy('((x,y),(x,z))')            # a matrix funciton in 2D
        v = str2sympy('sin(x)*sin(y)')            # a scalar function in 2D
    """
    exp = sympify(str)
    if isinstance(exp, (tuple, list)):
        return Matrix(exp)
    else:
        return exp


def sympy2exp(exp):
    """
    Convert a sympy expression to FEniCS expression. For example,
        u = str2sympy('sin(x)*sin(y)')
        f = Expression(Grad(u))
    creates a FEniCS Expression whose value is the gradient of sin(x)*sin(y).
    """
    x, y, z = symbols('x[0] x[1] x[2]')

    def to_ccode(f):
        f = f.subs('x', x).subs('y', y).subs('z', z)
        raw = printing.ccode(f)
        return raw.replace('M_PI', 'pi')
    if hasattr(exp, '__getitem__'):
        if exp.shape[0] == 1 or exp.shape[1] == 1:
            return tuple(map(to_ccode, exp))
        else:
            return tuple([tuple(map(to_ccode, exp[i, :]))
                          for i in range(exp.shape[1])])
    else:
        return to_ccode(exp)


def _infer_dim(u):
    atoms = u.atoms()
    if sympify('z') in atoms:
        return 3
    elif sympify('y') in atoms:
        return 2
    else:
        return 1


def Grad(u, dim=None):
    """
    Scalar, vector, or matrix gradient.
    If dim is not given, the dimension is inferred. For exmaple,
        v = str2sympy('sin(x)*sin(y)')
        f0 = Grad(v)           # only (x,y). infer as 2D. f0 is a 2D vector.
        f1 = Grad(v, dim = 2)  # specified as 2D. f1 is a 2D vector.
        f2 = Grad(v, dim = 3)  # specified as 3D. f2 is a 3D vector.
    """
    if not dim:
        dim = _infer_dim(u)
    # transpose first if it is a row vector
    if u.is_Matrix and u.shape[0] != 1:
        u = u.transpose()
    # take the gradient
    if dim == 1:
        return Matrix([u.diff('x')]).transpose()
    elif dim == 2:
        return Matrix([u.diff('x'), u.diff('y')]).transpose()
    elif dim == 3:
        return Matrix(
            [u.diff('x'), u.diff('y'), u.diff('z')]).transpose()


def Curl(u):
    """Vector curl in 2D and 3D."""
    if u.is_Matrix and min(u.args) == 1:
        # 3D vector curl
        return Matrix([u[2].diff('y') - u[1].diff('z'),
                       u[0].diff('z') - u[2].diff('x'),
                       u[1].diff('x') - u[0].diff('y')])
    else:
        # 2D rotated gradient
        return Matrix([u.diff('y'), -u.diff('x')])


def Rot(u):
    """Vector rot in 2D. The result is a scalar function."""
    # 2d rot
    return u[1].diff('x') - u[0].diff('y')


def Div(u):
    """Vector and matrix divergence. For matrices, the divergence is taken
       row-by-row."""
    def vec_div(w):
        if w.shape[0] == 2:
            return w[0].diff('x') + w[1].diff('y')
        elif w.shape[0] == 3:
            return w[0].diff('x') + w[1].diff('y') + w[2].diff('z')
    if u.shape[1] == 1 and len(u.shape) == 2:
        # column vector
        return vec_div(u)
    elif u.shape[0] == 1 and len(u.shape) == 2:
        # row vector
        return vec_div(u.transpose())
    else:
        # matrix
        result = []
        for i in range(u.shape[1]):
            result.append(vec_div(u.row(i).transpose()))
        return Matrix(result)


def Sym(u):
    """Matrix symmetrization."""
    return (u + u.transpose()) / 2.0


def Tr(u):
    """Matrix trace."""
    return u.trace()


def Hess(u, dim=None):
    """The Hessian."""
    return Grad(Grad(u, dim), dim)


def HodgeStar(u):
    """Unweighted Hodge star in Euclidean basis in 2D and 3D.
    In 2D, it rotates a vector counterclockwise by pi/2:
       [u0, u1] -> [-u1, u0]
    In 3D, it maps a vector to an antisymmetric matrix:
                       [0  -u2  u1]
       [u0, u1, u2] -> [ u2 0  -u0]
                       [-u1 u0  0 ]
    and it maps an antisymmetric matrix back to a vector reversing the above.
    """
    if len(u) == 2:
        # 2D
        return Matrix((-u[1], u[0]))
    elif len(u) == 3:
        # 3D
        if u.shape[0] * u.shape[1] == 3:
            # vector
            return Matrix(((0, -u[2], u[1]),
                           (u[2], 0, -u[0]),
                           (-u[1], u[0], 0)))
        else:
            # matrix
            if u.transpose() == -u:
                return Matrix((u[2, 1], u[0, 2], u[1, 0]))
            else:
                raise RuntimeError("Input matrix for Hodge star is not"
                                   "anti-symmetric.")


def Epsilon(u):
    """Vector symmetric gradient."""
    return Sym(Grad(u.transpose()))

Eye = eye
