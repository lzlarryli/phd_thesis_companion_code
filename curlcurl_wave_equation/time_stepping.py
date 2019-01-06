"""PDE time stepping module."""

from fenics import (Function, TrialFunction, TestFunction, dot, dx, action,
                    LinearVariationalProblem, LinearVariationalSolver,
                    NonlinearVariationalProblem, NonlinearVariationalSolver,
                    derivative, interpolate)

def leapfrog(A, f, ah, bh, k, T):
    """Leapfrog time-stepper.

    Abstract problem:

        u'' + Au = f,  u(0) = ah, u'(0) = bh

    Args:
        A (Function -> Function -> Form): Possibly nonlinear functional A(u, v)
        f (float -> Function): Right-hand side
        ah (Function): Initial value
        bh (Function): Initial velocity
        k (float): Time step
        T (float): Max time

    Returns:
        list(float): Times
        list(Function): Solution at each time
    """
    # pylint: disable=invalid-name, too-many-arguments, too-many-locals
    # Basic setup
    V = ah.function_space()
    u = TrialFunction(V)
    v = TestFunction(V)
    # Prepare initial condition
    u0 = Function(V)
    u0.vector()[:] = ah.vector()
    u1 = Function(V)
    u1.vector()[:] = ah.vector() + k * bh.vector()
    # Initialize time stepper
    t = k
    u2 = Function(V)
    u2.vector()[:] = u1.vector()
    ts = [0, t]
    uh = [interpolate(u0, V), interpolate(u1, V)]
    progress = -1
    print("Progress: ", end="")
    while t < T:
        # Print progress
        pct = int(t / T * 100) // 10 * 10
        if pct > progress:
            print("{}%..".format(pct), end="")
            progress = pct
        # Solve
        lhs = dot(u, v) * dx
        rhs = (dot(2 * u1 - u0, v) + k * k * (f(t, v) - A(u1, v))) * dx
        problem = LinearVariationalProblem(lhs, rhs, u2)
        solver = LinearVariationalSolver(problem)
        solver.parameters["linear_solver"] = "cg"
        solver.parameters["preconditioner"] = "hypre_amg"
        try:
            solver.solve()
        except RuntimeError:
            print("blowup at t={}.".format(t))
            return (ts, uh)
        # Update
        t += k
        u0.vector()[:] = u1.vector()
        u1.vector()[:] = u2.vector()
        # Record result
        ts.append(t)
        uh.append(interpolate(u1, V))
    print("done")
    return (ts, uh)


def theta_method(A, f, ah, bh, θ, k, T):
    """θ-method time-stepper.

    Abstract problem:

        u'' + Au = f,  u(0) = ah, u'(0) = bh

    is formulated as a first-order system:

        u' - v = 0,
        v' + Au = f,
        u(0) = ah, v(0) = bh.

    Args:
        A (Function -> Function -> Form): Possibly nonlinear functional A(u, v)
        f (float -> Function): Right-hand side
        ah (Function): Initial value
        bh (Function): Initial velocity
        k (float): Time step
        T (float): Max time

    Returns:
        list(float): Times
        list(Function): Solution at each time
    """
    # pylint: disable=invalid-name, too-many-arguments, too-many-locals
    # Basic setup
    V = ah.function_space()
    u = TrialFunction(V)
    w = TestFunction(V)
    v = TrialFunction(V)
    y = TestFunction(V)
    α = k * θ
    β = k * (1.0 - θ)
    # Prepare initial condition
    u0 = Function(V)
    u0.vector()[:] = ah.vector()
    v0 = Function(V)
    v0.vector()[:] = bh.vector()
    # Initialize time stepper
    t = k
    u1 = Function(V)
    u1.vector()[:] = u0.vector()
    v1 = Function(V)
    v1.vector()[:] = v0.vector()
    ts = [0]
    uh = [interpolate(u0, V)]
    progress = -1
    print("Progress: ", end="")
    while t < T:
        # Print progress
        pct = int(t / T * 100) // 10 * 10
        if pct > progress:
            print("{}%..".format(pct), end="")
            progress = pct
        # Solve for the next u (nonlinear)
        lhs = (dot(u, w) + α ** 2 * A(u, w)) * dx
        rhs = (dot(u0, w) - α * β * A(u0, w) + k * dot(v0, w)
               + α ** 2 * f(t + k, w) + α * β * f(t, w)) * dx
        act = action(lhs - rhs, u1)
        J = derivative(act, u1)
        problem = NonlinearVariationalProblem(act, u1, [], J)
        solver = NonlinearVariationalSolver(problem)
        solver.parameters["newton_solver"]["linear_solver"] = "gmres"
        solver.parameters["newton_solver"]["preconditioner"] = "ilu"
        try:
            solver.solve()
        except RuntimeError:
            print("blowup at t={}.".format(t))
            return (ts, uh)
        # Solve for the next v (linear)
        lhs = dot(v, y) * dx
        rhs = (dot(u1 - u0, y) - β * dot(v0, w)) / α * dx
        problem = LinearVariationalProblem(lhs, rhs, v1)
        solver = LinearVariationalSolver(problem)
        solver.parameters["linear_solver"] = "cg"
        solver.parameters["preconditioner"] = "hypre_amg"
        solver.solve()
        # Update
        t += k
        u0.vector()[:] = u1.vector()
        v0.vector()[:] = v1.vector()
        # Record result
        ts.append(t)
        uh.append(interpolate(u1, V))
    print("done")
    return (ts, uh)
