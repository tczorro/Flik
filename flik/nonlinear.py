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


r"""
Solvers for nonlinear systems using the Newton and Gauss-Newton algorithms.

Functions used to find the roots of a nonlinear system of equations given the
residual function `f` and its analytical Jacobian `J`. An exactly-determined
system (`m` equations, `m` variables) is best solved with the Newton method.
Over- or under- determined systems (`m` equations, `n` variables) must be
solved in the least-squares sense using the Gauss-Newton method.

"""


from numbers import Integral
from numbers import Real

import numpy as np

from flik.jacobian import Jacobian
from flik.approx_jacobian import CentralDiffJacobian


__all__ = [
    "nonlinear_solve",
    ]


def nonlinear_solve(f, x_0, J=None, stepsize=1.0, eps=1.0e-6, maxiter=100, method="newton"):
    r"""
    Solve a system of nonlinear equations with the Newton method.

    Parameters
    ----------
    f : callable
        Vector-valued function corresponding to nonlinear system of equations.
        Must be of the form f(x), where x is a 1-dimensional array.
    x_0 : np.ndarray
        Solution initial guess.
    J : callable, optional
        Jacobian of function f. Must be of the form J(x), where x is a
        1-dimensional array. If none is given, then the Jacobian is calculated
        using finite differences.
    stepsize : float or np.ndarray, optional
        Scaling factor for Newton step.
    eps : float, optional
        Convergence threshold for vector function f norm.
    maxiter : int, optional
        Maximum number of iterations to perform.
    method : str, optional
        Update method for the (approximated) J(x) or the inverse of J(x). The
        default uses Newton method.

    Returns
    -------
    result : dict
        A dictionary with the keys:
        success
            Boolean variable informing whether the algorithm succeeded or not.
        message
            Information about the cause of the termination.
        niter
            Number of actual iterations performed.
        x
            Nonlinear system of equations solution (Root).
        f
            Vector function evaluated at solution.
        J
            Jacobian evaluated at solution.
        eps
            Convergence threshold for vector function f norm.

    Raises
    ------
    TypeError
        If an argument of an invalid type or shape is passed.
    ValueError
        If an argument passed has an unreasonable value.

    """
    # Check input types and values
    if not callable(f):
        raise TypeError("Argument f should be callable")
    if not callable(J) and J is not None:
        raise TypeError("Argument J should be callable")
    if not (isinstance(x_0, np.ndarray) and x_0.ndim == 1):
        raise TypeError("Argument x_0 should be a 1-dimensional numpy array")
    if not isinstance(eps, Real):
        raise TypeError("Argument eps should be a real number")
    if not isinstance(maxiter, Integral):
        raise TypeError("Argument maxiter should be an integer number")
    if not isinstance(method, str):
        raise TypeError("Argument method should be a string")
    if eps < 0.0:
        raise ValueError("Argument eps should be >= 0.0")
    if maxiter < 1:
        raise ValueError("Argument maxiter should be >= 1")
    eps = float(eps)
    maxiter = int(maxiter)
    # Check stepsize argument
    if isinstance(stepsize, Real):
        stepsize = float(stepsize)
    elif isinstance(stepsize, np.ndarray):
        if stepsize.shape != x_0.shape:
            raise TypeError("stepsize and x_0 must have the same shape")
    else:
        raise TypeError("Argument stepsize should be a float or numpy array")
    if np.any(stepsize < 0.0):
        raise ValueError("Argument stepsize should be >= 0.0")
    # Check J (Jacobian function) argument
    if J is None:
        m = f(x_0).shape[0]
        n = x_0.shape[0]
        J = CentralDiffJacobian(f, m, n)
    else:
        J = J if isinstance(J, Jacobian) else Jacobian(J)
    # Choose the step/update function and inverse option
    inverse, step, update = _nonlinear_functions(J, method)
    # Return result of Newton iterations
    return _nonlinear_iterations(f, x_0, J, stepsize, eps, maxiter, inverse, step, update)


def _nonlinear_functions(J, method):
    r"""
    Return the functions used in ``nonlinear_solve`` according to the method.

    Parameters
    ----------
    J : Jacobian
    method : str

    Returns
    -------
    inv : bool
        True if inverse Jacobian is used, otherwise False
    step : function
        Newton step function
    update : function
        Jacobian update function

    """
    method = method.lower()
    if method == "newton":
        result = False, _step_linear, J.update_newton
    elif method == "goodbroyden":
        result = False, _step_linear, J.update_goodbroyden
    elif method == "dfp":
        result = False, _step_linear, J.update_dfp
    elif method == "sr1":
        result = False, _step_linear, J.update_sr1
    elif method == "badbroyden":
        result = True, _step_inverse, J.update_badbroyden
    elif method == "bfgs":
        result = True, _step_inverse, J.update_bfgs
    elif method == "sr1inv":
        result = True, _step_inverse, J.update_sr1inv
    elif method == "gaussnewton":
        result = False, _step_gauss_newton, J.update_newton
    else:
        raise ValueError("Argument method is not a valid option")
    return result


def _nonlinear_iterations(f, x_0, J, stepsize, eps, maxiter, inverse, step, update):
    r"""Run the iterations for ``newton_solve``."""
    # Calculate f_0 and J_0
    A = np.linalg.inv(J(x_0)) if inverse else J(x_0)
    b = f(x_0)
    # Iterations
    success = False
    message = "Maximum number of iterations reached."
    for niter in range(1, maxiter + 1):
        b *= -1
        # Calculate step function, take Newton step
        try:
            dx = step(b, A)
            dx *= stepsize
        except np.linalg.LinAlgError:
            message = "Singular Jacobian; no solution found."
            break
        x_0 += dx
        # Evaluate function and Jacobian for next step or result
        df = b
        b = f(x_0)
        df += b
        update(A, x_0, dx, df)
        # Check for convergence
        if np.linalg.norm(b) < eps:
            # If so, we're done (SUCCESS)
            success = True
            message = "Convergence obtained."
            break
    # Return result dictionary
    return {
        "success": success,
        "message": message,
        "niter": niter,
        "x": x_0,
        "f": b,
        "J": np.linalg.inv(A) if inverse else A,
        "eps": eps,
        }


def _step_linear(b, A):
    r"""
    Compute the Newton step for the Jacobian.

    Calculate the roots for the next step of a method that updates the
    (approximated) Jacobian matrix.

    Parameters
    ----------
    b : np.ndarray
        1-dimensional array of the negative of the function evaluated at the
        current guess of the roots -f(x_0).
    A : np.ndarray
        2-dimensional array of the Jacobian evaluated at the current guess of
        the roots J(x_0).

    Returns
    -------
    dx : np.ndarray
        Step length for the method.

    """
    return np.linalg.solve(A, b)


def _step_inverse(b, A):
    r"""
    Compute the Newton step for the inverse of the Jacobian.

    Calculate the roots for the next step of a method that updates the
    inverse of the approximated Jacobian matrix.

    Parameters
    ----------
    b : np.ndarray
        1-dimensional array of the negative of the function evaluated at the
        current guess of the roots -f(x_0).
    A : np.ndarray
        2-dimensional array of the Jacobian evaluated at the current guess of
        the roots J(x_0).

    Returns
    -------
    dx : np.ndarray
        Step length for the method.

    """
    return np.dot(A, b)


def _step_gauss_newton(b, A):
    r"""
    Compute the Gauss-Newton step for the Jacobian.

    Calculate the roots for the next step of a method that updates the
    (approximated) Jacobian matrix.

    Parameters
    ----------
    b : np.ndarray
        1-dimensional array of the negative of the function evaluated at the
        current guess of the roots -f(x_0).
    A : np.ndarray
        2-dimensional array of the Jacobian evaluated at the current guess of
        the roots J(x_0).

    Returns
    -------
    dx : np.ndarray
        Step length for the method.

    """
    return np.linalg.solve(np.dot(A.T, A), np.dot(A.T, b))
