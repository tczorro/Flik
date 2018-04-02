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
NUM_ITERS = 30


def quasi_newtons_opt(function, gradient, hessian, val,
                      update=None, inv=False,
                      convergence=CONV,
                      num_iterations=NUM_ITERS):
    """Quasi-Newton method for approximating hessians.

    Parameters
    ----------
    function : Callable
        The function for which the minimum will
        be computed.
        returns np.array((1))
    gradient: Callable
        The gradientient of the given function.
        (1st derivative)
        returns np.array((N, ))
    hessian : Callable
        The approximation to hessian of the given function.
        (2nd derivative)
        returns np.array((N, N))
    val : numpy.ndarray((N, ))
        An initial guess for the function's
        minimum value.
    update : Callable
        Hessian approximation method.
        If none just newtons method.
        returns np.array((N, N))
    inv : bool
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
    # Check input
    if not callable(function):
        raise TypeError('Function should be callable')
    if not callable(gradient):
        raise TypeError('Gradient should be callable')
    if not callable(hessian):
        raise TypeError('Gradient should be callable')
    if not (isinstance(val, np.ndarray) and val.ndim == 1):
        raise TypeError("Argument val should be a 1-dimensional numpy array")
    if not isinstance(convergence, Real):
        raise TypeError("Argument convergence should be a real number")
    if not isinstance(num_iterations, Integral):
        raise TypeError("Argument num_iterations should be an integer number")
    if convergence < 0.0:
        raise ValueError("Argument convergence should be >= 0.0")
    if num_iterations <= 0:
        raise ValueError("Argument num_iterations should be >= 1")
    if not isinstance(inv, bool):
        raise TypeError("Inverse option is either True or False")
    # choose initial guess and non-singular hessian approximation
    point = val
    hess = hessian(point)
    step_direction = None

    # non-optimized step length
    step_length = 1

    for i in range(1, num_iterations+1):
        if not update:
            hess = hessian(point)

        # calculate step
        if len(point) > 1:
            # Check if hessian is approximated as inverse
            if inv and (i > 1):
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
    raise ValueError("Ran out of iterations.")


def update_hessian_bfgs(hessian, gradient, point, point1, inv=False):
    """Bfgs update for quasi-newton.

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
    point : np.ndarray
        An array representing the point at which
        to evaluate the function, gradient, and
        hessian.
    point1 : np.ndarray
        An array representing the next point for
        evaluation (k plus 1).
    inv : bool, default False
        If True, the given hessian is inverted, and
        the resulting hessian should be calculated
        accordingly. If False, the given hessian is
        not inverted, and the BFGS method should be
        applied as normal.

    Returns
    -------
    np.ndarray
        The estimation of the hessian (evaluated).

    """
    sk = point1 - point
    yk = gradient(point1) - gradient(point)
    newhess = hessian.copy()
    # regular hessian approximation
    if not inv:
        # part one
        one = np.dot(np.dot(newhess, np.outer(sk, sk)), newhess)
        newhess -= one
        # part two
        newhess += np.outer(yk, yk)/np.dot(yk, sk)
    # inverted hessian approximation
    else:
        # can change this to gradient.shape() if
        # the input is guaranteed to be a numpy array
        size = len(gradient)
        print('size', size)
        # denominator (to avoid calculating 3 times)
        denom = np.dot(yk, sk)
        # part one
        one = np.identity(size) - np.outer(sk, yk)/denom
        # part two
        two = np.identity(size) - np.outer(yk, sk)/denom
        # part three
        newhess = np.dot(np.dot(one, hessian), two) + np.outer(sk, sk)/denom
    return newhess


def update_hessian_broyden(hessian, gradient, point, point1, inv=False):
    """Approximate Hessian with new x Good Broyden style.

    Parameters
    ----------
    hessian : np.ndarray((N, N))
    gradient : Callable
    point : np.array((N, ))
    point1 : np.array((N, ))
    inv : str

    Returns
    -------
    hessian : np.ndarray((N, N))

    """
    sk = point1 - point
    yk = gradient(point1) - gradient(point)
    # approximate inverse Hessian
    if inv:
        t = np.dot(np.outer(sk - np.dot(hessian, yk), sk), hessian)
        t /= np.outer(np.dot(sk, hessian), yk)
        hessian += t
    # approximate Hessian
    else:
        yk -= np.dot(hessian, sk)
        yk /= np.dot(sk, sk)
        hessian += np.outer(yk, sk.T)
    return hessian


def update_hessian_sr1(hessian, gradient, point, point1,
                       inv=False):
    """Approximate Hessian with new SR1 style.

    Parameters
    ----------
    hessian : np.ndarray((N, N))
        An evaluated hessian (square matrix).
    gradient : Callable
        The gradient of the function (not
        evaluated).
    point : np.array((N, ))
        An array representing the point at which
        to evaluate the function, gradient, and
        hessian.
    point1 : np.array((N, ))
        An array representing the next point for
        evaluation (k plus 1).
    inv : bool, default False
        If True, calculates the result as if
        the given evaluated hessian was inverted.
        If False, calculates the result as normal.

    Returns
    -------
    hessian : np.ndarray((N, N))
        The estimation of the hessian (evaluated).

    """
    sk = point1 - point
    yk = gradient(point1) - gradient(point)
    newhess = hessian.copy()
    if not inv:
        # name piece not important
        piece = yk - np.dot(newhess, sk)
        piece = np.outer(piece, piece) / np.dot(piece, sk)
        newhess += piece
    else:
        piece = sk - np.dot(newhess, yk)
        newhess += np.outer(piece, piece)/np.dot(piece, yk)
    return newhess


def update_hessian_dfp(hessian, gradient, point, point1,
                       inv=False):
    """Approximate Hessian with new x DFP style.

    Parameters
    ----------
    hessian : np.ndarray((N, N))
    gradient : Callable
    point : np.array((N, ))
    point1 : np.array((N, ))
    inv : str

    Returns
    -------
    hessian : np.ndarray((N, N))

    """
    sk = point1 - point
    yk = gradient(point1) - gradient(point)
    newhess = hessian.copy()
    # approximate inverse Hessian
    if inv:
        a = np.outer(sk, sk) / np.dot(sk, yk)
        b = np.dot(np.dot(newhess, np.outer(yk, yk)), newhess)
        b /= np.dot(np.dot(yk, newhess), yk)
        newhess = newhess + a - b
    # approximate Hessian
    else:
        a = -np.outer(yk, sk) / np.dot(yk, sk)
        b = -np.outer(sk, yk) / np.dot(yk, sk)
        c = np.outer(yk, yk) / np.dot(yk, sk)
        a += np.eye(a.shape[0])
        b += np.eye(b.shape[0])
        newhess = np.dot(np.dot(a, newhess), b) + c
    return newhess
