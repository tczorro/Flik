#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""Newton's method file from flik.

This module contains one function that runs Newton's method.

Example
-------
Import this module and run the function on the function of
your choice. You will need to know the gradient (first
derivative), hessian (second derivative), and an initial guess.

    from flik.minimization import newtons_method

    # using default arguments
    results = newtons_method(func, gradient, hessian,
    init_point)

    # using user-specified arguments
    results = newtons_method(funct, gradient, hessian,
    init_point, convergence=0.001, num_iterations=50)

Notes
-----
You can include optional arguments to specify the maximum
number of iterations to perform (num_iterations) and the
convergence tolerance (convergence). Convergence tolerance
is the maximum difference permitted between the gradient's
value at the calculated minimum/maximum and zero.


Attributes
----------
CONV : float
    Default convergence condition is 10E-06.

NUM_ITERS : int
    Default number of iterations to do is 100.


.. _flik Documentation HOWTO:
    https://github.com/QuantumElephant/Flik

"""

import numpy as np

CONV = 10E-06
NUM_ITERS = 100


def newtons_opt(function, gradient, hessian, initial_point,
                convergence=CONV, num_iterations=NUM_ITERS):
    """Perform Newton's method for a given function.

    Parameters
    ----------
    function : Callable
        The function for which the minimum will
        be computed.
    gradient : Callable
        The gradient of the given function.
        (1st derivative)
    hessian : Callable
        The hessian of the given function.
        (2nd derivative)
    initial_point : numpy.ndarray
        An initial guess for the function's
        minimum value.
    convergence : float
        The condition for convergence (acceptable
        error margin from zero for the returned
        minimum).
    num_iterations : int
        The maximum number of iterations to do
        in order to reach convergence.

    Notes
    -----
    Cannot perform inversion (np.linalg.inv) on a single
    function (needs to be at least a 2x2 array).
    The function, np.allclose, returns True if two arrays
    are element-wise equal within a tolerance.

    Returns
    -------
    Tuple (
        function(x) : numpy.ndarray
            The value of the function at the
            minimum found within the convergence
            condition.
        x : numpy.ndarray
            The coordinates that represent the
            minimum of the function.
        gradient(x) : numpy.ndarray
            The value of the gradient at the
            minimum found.
        i : int
            How many iterations were needed
            for convergence.
    )

    Raises
    ------
    ValueError
        If the function does not converge to within
        the given convergence value in the number
        of iterations specified by num_iterations.

    """
    min_point = initial_point
    for i in range(1, num_iterations+1):
        # dot product between inverse(hessian) and gradient
        if len(initial_point) > 1:
            min_point = min_point - np.dot(gradient(min_point), np.linalg.inv(hessian(min_point)))
        # only single variable function
        else:
            min_point = min_point - np.dot(gradient(min_point), 1/hessian(min_point))
        if np.allclose(gradient(min_point), 0, atol=convergence):
            return (function(min_point), min_point, gradient(min_point), i)
    raise ValueError(f"The function didn't converge within {num_iterations} iterations.")
