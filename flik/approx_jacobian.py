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
Classes for numerical approximations for Jacobians of analytical functions.

Numerical approximations to the Jacobian are useful for optimization purposes
where the analytical Jacobian is unavailable or prohibitively expensive to
compute.

The forward difference Jacobian approximation uses the formula:
..math:: \frac{\partial f_i(x)}{\partial x_j}
            = \frac{f(x + \epsilon e_j) - f(x)}{\epsilon}

The central difference Jacobian approximation uses the formula:
..math:: \frac{\partial f_i(x)}{\partial x_j}
            = \frac{f(x + \epsilon e_j) - f(x - \epsilon e_j)}{2 \epsilon}

where :math: `e_j` is the unit vector in dimension :math: `j` and :math:
`\epsilon` is a small finite increment over which to approximate the Jacobian.

"""


from numbers import Integral
from numbers import Real

import numpy as np


__all__ = [
    "FiniteDiffJacobian",
    "ForwardDiffJacobian",
    "CentralDiffJacobian",
    ]


class FiniteDiffJacobian:
    r"""
    Finite difference Jacobian approximation class.

    A callable object that approximates the Jacobian of a function `f` using
    a finite difference approximation.

    """

    def __init__(self, f, m, n=None, eps=1.0e-4):
        r"""
        Construct a finite difference approximate Jacobian function.

        Parameters
        ----------
        f : callable
            The function for which the Jacobian is being approximated.
        m : int
            Size of the function output vector.
        n : int, optional
            Size of the function argument vector (default is `n` == `m`).
        eps : float or np.ndarray, optional
            Increment in the function's argument to use when approximating the
            Jacobian.

        Raises
        ------
        TypeError
            If an argument of an invalid type or shape is passed.
        ValueError
            If an argument passed has an unreasonable value.

        """
        # Handle default arguments
        if n is None:
            n = m

        # Check input types and values
        if not callable(f):
            raise TypeError("f must be a callable object")
        if not isinstance(m, Integral):
            raise TypeError("m must be an integral type")
        if not isinstance(n, Integral):
            raise TypeError("n must be an integral type")
        if not (isinstance(eps, np.ndarray) and eps.ndim == 1):
            if not isinstance(eps, Real):
                raise TypeError("eps must be a float or 1-dimensional array")
        if m <= 0:
            raise ValueError("m must be > 0")
        if n <= 0:
            raise ValueError("n must be > 0")
        if isinstance(eps, np.ndarray):
            if eps.size != n:
                raise ValueError("eps must be of the same length as the input vector")
            eps = np.copy(eps)
        else:
            eps = np.full(int(n), float(eps), dtype=np.float)
        if np.any(eps <= 0.0):
            raise ValueError("eps must be > 0.0")

        # Assign internal attributes
        self._function = f
        self._m = int(m)
        self._n = int(n)
        self._eps = eps


class ForwardDiffJacobian(FiniteDiffJacobian):
    r"""
    Forward difference Jacobian approximation class.

    A callable object that approximates the Jacobian of a function `f` via the
    following formula:
    ..math:: \frac{\partial f_i(x)}{\partial x_j}
                = \frac{f(x + \epsilon e_j) - f(x)}{\epsilon}

    """

    def __call__(self, x, fx=None):
        r"""
        Evaluate the approximate Jacobian at position `x`.

        Parameters
        ----------
        x : np.ndarray
            Argument vector to the approximate Jacobian function.
        fx : np.ndarray, optional
            Output vector of the function at position `x` (optional, but avoids
            an extra function call).

        Returns
        -------
        jacobian : np.ndarray
            Value of the approximate Jacobian at position `x`.

        """
        # Note: In order to stick to row-major iteration, this algorithm
        # computes the transpose of the approximate Jacobian into the jac
        # vector. This function, being the Jacobian proper, returns the
        # transpose of the jac vector.
        jac = np.empty((self._n, self._m), dtype=np.float)

        # Evaluate function at x (fx = f(x)) if required
        if fx is None:
            fx = self._function(x)

        # Copy x to vector dx
        dx = np.copy(x)

        # Iterate over elements of `x` to increment
        for i in range(self._n):
            # Add forward-epsilon increment to dx (dx = x + e_i * eps_i)
            dx[i] += self._eps[i]
            # Evaluate function at dx (dfx = f(dx))
            dfx = self._function(dx)
            # Calculate df[j]/dx[i] = (dfx - fx) / eps_i into dfx vector
            dfx -= fx
            dfx /= self._eps[i]
            # Put result from dfx into the ith row of the jac matrix
            jac[i, :] = dfx
            # Reset dx = x
            dx[i] = x[i]

        # df[i]/dx[j] = transpose(jac)
        return jac.transpose()


class CentralDiffJacobian(FiniteDiffJacobian):
    r"""
    Central difference Jacobian approximation class.

    A callable object that approximates the Jacobian of a function `f` via the
    following formula:

    ..math:: \frac{\partial f_i(x)}{\partial x_j}
                = \frac{f(x + \epsilon e_j) - f(x - \epsilon e_j)}{2 \epsilon}

    """

    def __call__(self, x):
        r"""
        Evaluate the approximate Jacobian at position `x`.

        Parameters
        ----------
        x : np.ndarray
            Argument vector to the approximate Jacobian function.

        Returns
        -------
        jacobian : np.ndarray
            Value of the approximate Jacobian at position `x`.

        """
        # Note: In order to stick to row-major iteration, this algorithm
        # computes the transpose of the approximate Jacobian into the jac
        # vector. This function, being the Jacobian proper, returns the
        # transpose of the jac vector.
        jac = np.empty((self._n, self._m), dtype=np.float)

        # Copy x to vector dx
        dx = np.copy(x)

        # Iterate over elements of `x` to increment
        for i in range(self._n):
            # Add forward-epsilon increment to dx (+dx = x + e_i * eps_i)
            dx[i] += self._eps[i]
            # Evaluate function at +dx (dfx2 = f(+dx))
            dfx2 = self._function(dx)
            # Add backward-epsilon increment to dx (-dx = x - e_i * eps_i)
            dx[i] = x[i] - self._eps[i]
            # Evaluate function at -dx (dfx1 = f(-dx))
            dfx1 = self._function(dx)
            # Calculate df[j]/dx[i] = (dfx2 - dfx1) / (2 * eps_i) into dfx2 vector
            dfx2 -= dfx1
            dfx2 /= 2 * self._eps[i]
            # Put result from dfx2 into the ith row of the jac matrix
            jac[i, :] = dfx2
            # Reset dx = x
            dx[i] = x[i]

        # df[i]/dx[j] = transpose(jac)
        return jac.transpose()
