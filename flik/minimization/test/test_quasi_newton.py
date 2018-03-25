"""Testing for newtons_method file from flik.

This module contains one function that tests newtons_method
for a single variable function, 5x^2 - 2x + 3.
It asserts that a single-variable function converges in one
step.

Example
-------
Run with nosetests or run from the terminal.

    $ nosetests3 flik
    $ python test_newtons_method.py


.. _flik Documentation HOWTO:
    https://github.com/QuantumElephant/Flik

"""

import numpy as np
from flik.minimization import quasi_newton
from flik.MultiVarFunction import MultiVarFunction, Gradient, Hessian
from numpy.testing import assert_almost_equal, assert_equal


def test_quasi_newton_quad1():
    """Test newtons_method for a single variable quadratic.

    Notes
    -----
    The test function is: 5x^2 - 2x + 3
    The test gradient is: 10x - 2
    The test hessian is: 10
    The test point is: [0.25]
    Expect result of newtons_method to be:
    (2.8, [0.2], 0.0, 0)

    Returns
    -------
    None

    """
    quad = MultiVarFunction({5: [2], -2: [1], 3: [0]}, 1)
    grad = MultiVarFunction({10: [1], -2: [0]}, 1)
    hess = MultiVarFunction({10: [0]}, 1)
    val = np.array([0.25])

    # check results function
    def success(res):
        assert_almost_equal(res[0], 2.8, decimal=4)
        assert_almost_equal(res[1], np.array([0.2]), decimal=4)
        assert_almost_equal(res[2], np.array([0.0]), decimal=4)

    # basically newton's method
    res = quasi_newton.quasi_newtons_opt(quad, grad, hess, val)
    success(res)
    # Broyden quasi-newton
    res = quasi_newton.quasi_newtons_opt(quad, grad, hess, val, quasi_newton.update_hessian_broyden)
    success(res)


def test_quasi_newton_quad2():
    """Test newtons_method for a 2 variable quadratic.

    Notes
    -----
    The test function is: 3x^2y - 7yx - 9
    The test gradient [df/dx, df/dy] is:
        [6xy-7y, 3x^2-7x]
    The test hessian ([d2f/dx2,  d2f/dxdy],
                      [d2f/dydx,  d2f/dy2]) is:
        [6y,  6x-7],
        [6x-7,   0]
    The test point is: [-6,0]
    Expect result of newtons_method to be:
    ?

    Returns
    -------
    None

    """
    # doesn't work yet
    jen = MultiVarFunction({3: [2, 1], -7: [1, 1], -9: [0, 0]}, 2)
    g = jen.construct_grad()
    h = jen.construct_hess()
    print(h([-6,0]))

    val = np.array([-6, 0])
    res = quasi_newton.quasi_newtons_opt(jen, g, h, val)

if __name__ == "__main__":
    test_quasi_newton_quad1()
    #test_quasi_newton_quad2()
