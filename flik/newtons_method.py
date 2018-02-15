#!/usr/bin/env python3


# JEN

# Feb 14, 2018

#TODO add type declaration syntax to function definition
#from typing import Tuple, Callable

import numpy as np

# define constants
# these are used as the default
# values for the newtons_opt function
CONV = 10E-06       # convergence condition (error)
NUM_ITERS = 100     # number of iterations to do


def newtons_opt(function, gradient, hessian, initial_point, convergence=CONV, num_iterations=NUM_ITERS):
    """Performs Newton's method for a given function in
    order to find the minimum.

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
    x = initial_point
    for i in range(num_iterations):
        # dot product between inverse(hessian) and gradient
        x = x - np.dot(gradient(x), np.linalg.inv(hessian(x))
        # np.allclose returns True if two arrays
        # are element-wise equal within a tolerance.
        if np.allclose(gradient(x), 0, atol=convergence):
            return (function(x), x, gradient(x), i)
    raise ValueError(f"The function didn't converge within {num_iterations} iterations.")

