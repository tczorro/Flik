# An experimental local optimization package
# Copyright (C) 2018 Ayers Lab <ayers@mcmaster.ca>.
#
# This file is part of Flik.
#
# Flik is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# Flik is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>


# API: method_name_B(B, secant, step): -> new_B
# or method_name_H(H, secant, step): -> new_H
# testfilename should be test_method_name.py in minimization/test


from numbers import Integral
from numbers import Real
import numpy as np

CONV = 10E-06
NUM_ITERS = 100

def quasi_newtons_opt(function, gradient, hessian, val, update=None,
                inv=None, convergence=CONV, num_iterations=NUM_ITERS):
    """Quasi-Newton method for approximate hessians.

    Parameters
    ----------
    function : Callable
        The function for which the minimum will
        be computed.
        returns np.array((1))
    gradient: Callable
        The gradientient of the given function.
        (1st derivative)
        returns np.array((N,))
    hessian : Callable
        The approximation to hessian of the given function.
        (2nd derivative)
        returns np.array((N,N))
    val : numpy.ndarray((N,))
        An initial guess for the function's
        minimum value.
    update : Callable
        Hessian approximation method.
        If none just newtons method.
        returns np.array((N,N))
    inv : str
        Option for inverse Hessian approximation.
        If none then approximate the Hessian.
    convergence : float
        The condition for convergence (acceptable
        error margin from zero for the returned
        minimum).
    num_iterations : int
        The maximum number of iterations to do
        in order to reach convergence.

    """
    #TODO: correct docs

    # Check input
    if not callable(function):
        raise TypeError('Fucntion should be callable')
    if not callable(gradient):
        raise TypeError('gradientient should be callable')
    if not callable(hessian):
        raise TypeError('gradientient should be callable')
    #if not (isinstance(val, np.ndarray) and val.ndim == 1):
    #    raise TypeError("Argument val should be a 1-dimensional numpy array")
    if not isinstance(convergence, Real):
        raise TypeError("Argument convergence should be a real number")
    if not isinstance(num_iterations, Integral):
        raise TypeError("Argument num_iterations should be an integer number")
    if convergence < 0.0:
        raise ValueError("Argument convergence should be >= 0.0")
    if num_iterations <= 0:
        raise ValueError("Argument num_iterations should be >= 1")

    # choose initial guess and non-singular hessian approximation
    point = val
    hess = hessian(point)

    # non-optimized step length
    step_length = 1

    for i in range(1, num_iterations+1):
        if not update:
            hess = hessian(point)

        # calculate step
        if len(point) > 1:
            # Check if hessian is approximated as inverse
            if inv:
                step_direction = np.dot(-gradient(point),
                                        hess)
            # Hessian is not inverse
            else:
                step_direction = -gradient(point).dot(
                    np.linalg.inv(hess))
        else:
            step_direction = -np.dot(gradient(point), 1/hess)

        # new x
        point1 = point + step_length * step_direction
        
        # new hessian callable by approximation
        if update:
            hess = update(hess, gradient, point, point1, inv)
        # update x
        point = point1
        
        # stop when minimum
        if np.allclose(gradient(point), 0, atol=convergence):
            return function(point), point, gradient(point), i

def update_hessian_bfgs(hessian, gradient, pointk, pointkp1, inv):
    """BFGs update for quasi-newton.

    Returns an estimate of the hessian as to avoid
    evaluating the hessian. Uses equation 6.19 from
    numerical optimisation book.

    Parameters
    ----------
    hessian : np.ndarray
        An evaluated hessian (square matrix).
    gradient : callable array
        The gradient of the function (not
        evaluated).
    pointk : np.ndarray
        An array representing the point at which
        to evaluate the function, gradient, and
        hessian.
    pointkp1 : np.ndarray
        An array representing the next point for
        evaluation (k plus 1).

    Returns
    -------
    np.ndarray
        The estimation of the hessian (evaluated).

    """
    sk = pointkp1 - pointk
    yk = gradient(pointk) - gradient(pointkp1)
    newhess = hessian.copy()
    # part one
    one = np.dot(np.dot(newhess, np.outer(sk, sk)), newhess)
    newhess -= one
    # part two
    newhess += np.outer(yk, yk)/np.dot(yk, sk)
    if inv:
        raise NotImplementedError
    return newhess

def update_hessian_broyden(hessian, gradient, point, point1, inv):
    """
    Approximate Hessian with new x
    Good Broyden style

    Parameters
    ----------
    hessian : np.ndarray((N,N))
    gradient : Callable
    point : np.array((N,))
    point1 : np.array((N,))
    
    Returns
    -------
    hessian : np.ndarray((N,N))
    """
    sk = point1 - point
    yk = gradient(point1) - gradient(point)
    # approximate inverse Hessian
    if inv:
        t = xk - np.dot(hessian, yk)
        t2 = np.outer(np.dot(sk, hessian), yk)
        hessian += np.dot(np.outer(t, sk), hessian) / t2
    # approximate Hessian
    else:
        yk -= np.dot(hessian, sk)
        yk /= np.dot(sk, sk)
        hessian += np.outer(yk, sk.T)
    return hessian

def update_hessian_sr1(hessian, gradient, point, point1):
    """
    Approximate Hessian with new x
    SR1 style

    Parameters
    ----------
    hessian : np.ndarray((N,N))
    gradient : Callable
    point : np.array((N,))
    point1 : np.array((N,))

    Returns
    -------
    hessian : np.ndarray((N,N))
    """
    sk = point1 - point
    yk = gradient(point1) - gradient(point)
    yk -= np.dot(hessian, sk)
    yk = np.outer(yk, yk.T) / np.outer(yk.T, sk)
    hessian += yk / sk
    if inv:
        raise NotImplementedError
    return hessian

def update_hessian_dfp(hessian, gradient, point, point1, inv):
    """
    Approximate Hessian with new x
    DFP style

    Parameters
    ----------
    hessian : np.ndarray((N,N))
    gradient : Callable
    point : np.array((N,))
    point1 : np.array((N,))

    Returns
    -------
    hessian : np.ndarray((N,N))
    """
    sk = point1 - point
    yk = gradient(point1) - gradient(point)
    # approximate inverse Hessian
    if inv:
        a = -np.outer(sk, yk) / np.dot(sk, yk)
        b = np.dot(np.dot(hessian, np.outer(yk, yk)), hessian)
        b /= np.outer(np.dot(yk,h),yk)
        hessian += a - b
    # spproximate Hessian
    else:
        a = -np.outer(yk, sk) / np.dot(yk, sk)
        b = -np.outer(sk, yk) / np.dot(yk, sk)
        c = -np.outer(yk, yk) / np.dor(yk, sk)
        a += np.eye(a.shape[0])
        b += np.eye(b.shape[0])
        hessian = np.dot(np.dot(a, hessian), b) + c
    return hessian
