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
    grad = quad.construct_grad()
    hess = quad.construct_hess()
    val = [0.25]
    res = quasi_newton.quasi_newtons_opt(quad, grad, hess, val)
    
    # check results
    assert res[0] == 2.8, "Incorrect value of test function at minimum."
    assert res[1] == [0.2], "Incorrect minimum found for test function."
    assert res[2] == 0.0, "Incorrect value for the gradient at the minimum found."
    assert res[3] == 1, "Incorrect number of iterations for the test function. Should be one."


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
    jen = MultiVarFunction({3: [2, 1], -7: [1, 1], -9: [0, 0]}, 2)
    g = jen.construct_grad()
    h = jen.construct_hess()
    print(h([-6,0]))

    val = [-6, 0]
    res = quasi_newton.quasi_newtons_opt(jen, g, h, val)
    print(res)
    '''
    # original function
    quad2 = MultiVarFunction({3: [2, 1], -7: [1, 1], -9: [0, 0]}, 2)
    # df/dx
    grad2a = MultiVarFunction({6: [1, 1], -7: [0, 1]}, 2)
    # df/dy
    grad2b = MultiVarFunction({3: [2, 0], -7: [1, 0]}, 2)
    # gradient
    grad2 = Gradient(np.array([grad2a, grad2b]))
    # d2f/dx2
    hess2a = MultiVarFunction({6: [0, 1]}, 2)
    # d2f/dxdy OR d2f/dydx
    hess2b = MultiVarFunction({6: [1, 0], -7: [0, 0]}, 2)
    # d2f/dy2
    hess2d = MultiVarFunction({0: [0, 0]}, 2)
    # hessian
    hess2 = Hessian(
        np.array(
            [Gradient([hess2a, hess2b]),
             Gradient([hess2b, hess2d])]
        )
    )
    # initial point
    val = [-6, 0]
    res = quasi_newton.quasi_newtons_opt(quad2, grad2, hess2, val)
    # positive number of iterations
    assert res[-1] > 0
    '''


if __name__ == "__main__":
    #test_quasi_newton_quad1()
    test_quasi_newton_quad2()
