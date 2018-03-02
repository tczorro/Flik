#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""Testing for newtons_method file from flik.

This module contains one function that tests newtons_method
for a single variable function, 5x^2 - 2x + 3.
It asserts that a single-variable function converges in one
step.

This module also contains a class MultiVarFunction that
inherits from the object class. The purpose of this object
is to provide callables that are evaluated for functions
at a given point.

Example
-------
Run with nosetests or run from the terminal.

    $ nosetests3 flik
    $ python test_newtons_method.py


.. _flik Documentation HOWTO:
    https://github.com/QuantumElephant/Flik

"""

import numpy as np
from flik.minimization import newtons_method


class MultiVarFunction(object):
    """Multiple variable function object.

    Parameters
    ----------
    structure : dict
        The keys are the values of the coefficients.
        The values of the keys is a list, where:
        structure[coeff1] = [x_order, y_order, z_order]
        Example: f(x,y) = axy^2 + bx^2 + cy + d
        structure[a] = [1, 2]
        structure[d] = [0, 0]
    num_var : int
        The number of variables in the function.

    Returns
    -------
    None

    """

    def __init__(self, structure, num_var):
        """Run constructor for MultiVarFunction class."""
        self.structure = structure
        self.num_var = num_var

    def __call__(self, point):
        """Evaluate the function at the given point.

        Parameters
        ----------
        point : list / numpy.array
            The point at which to evaluate the function.
            The order of evaluation should be the same
            as for the dictionary structure.
            Example:
            f(x,y,z) = 10x^2y^3z^1
            point = [x,y,z]
            structure = { 10: [2,3,1] }

        Returns
        -------
        float
            The value of the function at the given point.

        Raises
        ------
        ValueError
            The length of the point list is not the same
            as the number of variables (num_var).

        """
        if len(point) != self.num_var:
            raise ValueError("Length of point list should be equal to the number of variables.")
        result = 0
        for coeff in self.structure:
            result += coeff*np.prod([point[i]**x for i, x in enumerate(self.structure[coeff])])
        return result


class Gradient(MultiVarFunction):
    """1D array of multiple variable function objects.

    Parameters
    ----------
    grad : list / np.array
        A list of MultiVarFunction objects.

    Returns
    -------
    None

    """

    def __init__(self, grad):
        """Run constructor for Gradient class."""
        self.grad = grad

    def __call__(self, point):
        """Evaluate the gradient at the given point.

        Parameters
        ----------
        point : list / numpy.array
            The point at which to evaluate the gradient.

        Returns
        -------
        np.array
            The value of the gradient at the given point.

        """
        return np.array([g(point) for g in self.grad])


class Hessian(Gradient):
    """2D array of multiple variable function objects.

    Parameters
    ----------
    grad : np.array
        A matrix containing Gradient objects.

    Returns
    -------
    None

    """

    def __init__(self, hess):
        """Run constructor for Hessian class."""
        self.hess = hess

    def __call__(self, point):
        """Evaluate the hessian at the given point.

        Parameters
        ----------
        point : list / numpy.array
            The point at which to evaluate the hessian.

        Returns
        -------
        np.array
            The value of the hessian at the given point.

        """
        return np.array([h(point) for h in self.hess])


def test_newtons_method_quad1():
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
    ip = [0.25]    # initial point
    res = newtons_method.newtons_opt(quad, grad, hess, ip)
    # check results
    assert res[0] == 2.8, "Incorrect value of test function at minimum."
    assert res[1] == [0.2], "Incorrect minimum found for test function."
    assert res[2] == 0.0, "Incorrect value for the gradient at the minimum found."
    assert res[3] == 1, "Incorrect number of iterations for the test function. Should be one."


def test_newtons_method_quad2():
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
    ip2 = [-6, 0]
    res = newtons_method.newtons_opt(quad2, grad2, hess2, ip2)
    # positive number of iterations
    assert res[-1] > 0


if __name__ == "__main__":
    test_newtons_method_quad1()
    test_newtons_method_quad2()
