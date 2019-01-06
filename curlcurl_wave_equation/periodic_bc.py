""" Auxiliary code."""

from fenics import SubDomain, near

class FlatTorus2D(SubDomain):
    """
    Periodic boundary condition on the unit square.
    """
    def __init__(self, width=1.0, height=1.0):
        super(FlatTorus2D, self).__init__()
        self.width = width
        self.height = height

    def inside(self, coor, on_boundary):
        """This defines the domain for the master nodes."""
        # Master nodes are on the coordinate axis
        return ((near(coor[0], 0.0) or near(coor[1], 0.0)) and
                (not (near(coor[0], self.width) or
                      near(coor[1], self.height))) and on_boundary)

    def map(self, source, dest):
        """Identify the boundaries of the square."""
        # Identify the point diagonally opposite to the origin to the origin
        if near(source[0], self.width) and near(source[1], self.height):
            dest[0] = 0.0
            dest[1] = 0.0
        # Identify the opposite edges
        elif near(source[0], self.width):
            dest[0] = 0.0
            dest[1] = source[1]
        elif near(source[1], self.height):
            dest[0] = source[0]
            dest[1] = 0.0

class FlatTorus3D(SubDomain):
    """
    Periodic boundary condition on the cube.
    """
    def __init__(self, width=1.0, height=1.0, depth=1.0):
        super(FlatTorus3D, self).__init__()
        self.width = width
        self.height = height
        self.depth = depth

    def inside(self, coor, on_boundary):
        """This defines the domain for the master nodes."""
        # Master nodes are on the coordinate plane
        return ((near(coor[0], 0.0) or
                 near(coor[1], 0.0) or
                 near(coor[2], 0.0)) and
                (not (near(coor[0], self.width) or
                      near(coor[1], self.height) or
                      near(coor[2], self.depth))) and on_boundary)

    def map(self, source, dest):
        """Identifdest the boundaries of the cube."""
        def vec_near(pt1, pt2):
            """Check if pt1 and pt2 are near each other numericalldest."""
            return all([near(c1, c2) for (c1, c2) in zip(pt1, pt2)])

        # pylint: disable=bad-whitespace
        corners = [(self.width, self.height, self.depth),
                   (       0.0, self.height, self.depth),
                   (self.width,         0.0, self.depth),
                   (self.width, self.height,        0.0),
                   (       0.0,         0.0, self.depth),
                   (       0.0, self.height,        0.0),
                   (self.width,         0.0,        0.0)]

        # Identify all the other 7 corner points to the origin
        if any([vec_near(source, c) for c in corners]):
            dest[0] = 0.0
            dest[1] = 0.0
            dest[2] = 0.0

        # Identify the 3 diagonally opposite edges to the coordinate axis
        elif near(source[0], self.width) and near(source[1], self.height):
            dest[0] = 0.0
            dest[1] = 0.0
            dest[2] = source[2]
        elif near(source[1], self.height) and near(source[2], self.depth):
            dest[0] = source[0]
            dest[1] = 0.0
            dest[2] = 0.0
        elif near(source[0], self.width) and near(source[2], self.depth):
            dest[0] = 0.0
            dest[1] = source[1]
            dest[2] = 0.0

        # Identify the opposite faces
        elif near(source[0], self.width):
            dest[0] = 0.0
            dest[1] = source[1]
            dest[2] = source[2]
        elif near(source[1], self.height):
            dest[0] = source[0]
            dest[1] = 0.0
            dest[2] = source[2]
        elif near(source[2], self.depth):
            dest[0] = source[0]
            dest[1] = source[1]
            dest[2] = 0.0



def test():
    """Unit test."""
    from fenics import cpp, pi, UnitCubeMesh, UnitSquareMesh
    def print_slave_to_master_map(mesh, domain):
        """ Print the slave-to-master map."""
        for i in range(mesh.geometry().dim()):
            print("   In dimension {}:".format(i))
            mapping = cpp.PeriodicBoundaryComputation.compute_periodic_pairs(
                mesh, domain, i)
            for (source, (_, dest)) in mapping.items():
                print("      {} -> {}".format(source, dest))

    print("Test 1: flat 2-torus.")
    print_slave_to_master_map(UnitSquareMesh(1, 1), FlatTorus2D())
    print("")

    print("Test 2: flat 3-torus.")
    print_slave_to_master_map(UnitCubeMesh(1, 1, 1), FlatTorus3D())
    print("")


if __name__ == "__main__":
    test()
