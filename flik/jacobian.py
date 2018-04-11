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


r"""Base Jacobian class."""


import numpy as np


__all__ = [
    "Jacobian",
    ]


class Jacobian:
    r"""
    Jacobian class with analytical evaluation by callable Jacobian function.

    The Jacobian class and its subclasses are used for evaluating and updating
    Jacobians as part of the Newton iterations.

    """

    def __init__(self, jac):
        r"""
        Construct a Jacobian class for a callable analytical jacobian.

        Parameters
        ----------
        jac : callable, optional

        Raises
        ------
        TypeError
            If an argument of an invalid type or shape is passed.

        """
        if not callable(jac):
            raise TypeError("J must be a callable object")
        self._jac = jac

    def __call__(self, x):
        r"""
        Compute the Jacobian at position ``x``.

        Parameters
        ----------
        x : np.ndarray

        Returns
        -------
        y :np.ndarray

        """
        return self._jac(x)

    def update_newton(self, A, new_x, *_):
        r"""
        Update the Jacobian matrix ``A`` at new solution vector ``x_(k+1)``.

        Parameters
        ----------
        new_x : np.ndarray
            ``x_(k+1)``
        dx : np.ndarray
            ``x_(k+1) - x_k``
        df : np.ndarray
            ``f_(k+1) - f_k``

        """
        A[...] = self(new_x)

    @staticmethod
    def update_goodbroyden(A, _, dx, df):
        r"""
        Update the Jacobian matrix ``A`` at new solution vector ``x_(k+1)``.

        Parameters
        ----------
        new_x : np.ndarray
            ``x_(k+1)``
        dx : np.ndarray
            ``x_(k+1) - x_k``
        df : np.ndarray
            ``f_(k+1) - f_k``

        """
        # Compute Good Broyden right hand side second term numerator
        t = df
        t -= np.dot(A, dx)
        # Divide by dx norm
        t /= np.dot(dx, dx)
        # Compute matrix from dot product of f and transposed dx
        A += np.outer(t, dx.T)

    @staticmethod
    def update_badbroyden(A, _, dx, df):
        r"""
        Update the Jacobian matrix ``A`` at new solution vector ``x_(k+1)``.

        Parameters
        ----------
        new_x : np.ndarray
            ``x_(k+1)``
        dx : np.ndarray
            ``x_(k+1) - x_k``
        df : np.ndarray
            ``f_(k+1) - f_k``

        """
        t2 = np.dot(dx.T, A)
        norm = np.dot(t2, df)
        t1 = dx
        t1 -= np.dot(A, df)
        t1 /= norm
        A += np.outer(t1, t2)

    @staticmethod
    def update_dfp(A, _, dx, df):
        r"""
        Update the Jacobian matrix ``A`` at new solution vector ``x_(k+1)``.

        Parameters
        ----------
        new_x : np.ndarray
            ``x_(k+1)``
        dx : np.ndarray
            ``x_(k+1) - x_k``
        df : np.ndarray
            ``f_(k+1) - f_k``

        """
        norm = np.dot(df, dx)
        t1 = np.outer(df, dx.T)
        t1 /= -norm
        t1 += np.eye(t1.shape[0])
        t2 = np.outer(dx, df.T)
        t2 /= -norm
        t2 += np.eye(t2.shape[0])
        A[:] = np.dot(t1, A)
        A[:] = np.dot(A, t2)
        t1 = np.outer(df, df.T)
        t1 /= norm
        A += t1

    @staticmethod
    def update_bfgs(A, _, dx, df):
        r"""
        Update the Jacobian matrix ``A`` at new solution vector ``x_(k+1)``.

        Parameters
        ----------
        new_x : np.ndarray
            ``x_(k+1)``
        dx : np.ndarray
            ``x_(k+1) - x_k``
        df : np.ndarray
            ``f_(k+1) - f_k``

        """
        Jacobian.update_dfp(A, None, df, dx)

    @staticmethod
    def update_sr1(A, _, dx, df):
        r"""
        Update the Jacobian matrix ``A`` at new solution vector ``x_(k+1)``.

        Parameters
        ----------
        new_x : np.ndarray
            ``x_(k+1)``
        dx : np.ndarray
            ``x_(k+1) - x_k``
        df : np.ndarray
            ``f_(k+1) - f_k``

        """
        t1 = df - np.dot(A, dx)
        t2 = np.outer(t1, t1.T)
        t2 /= np.dot(t1.T, dx)
        A += t2

    @staticmethod
    def update_sr1inv(A, _, dx, df):
        r"""
        Update the Jacobian matrix ``A`` at new solution vector ``x_(k+1)``.

        Parameters
        ----------
        new_x : np.ndarray
            ``x_(k+1)``
        dx : np.ndarray
            ``x_(k+1) - x_k``
        df : np.ndarray
            ``f_(k+1) - f_k``

        """
        Jacobian.update_sr1(A, None, df, dx)
