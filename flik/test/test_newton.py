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


"""Test file for `flik.nonlinear.newton_solve`."""


import numpy as np

from nose.tools import assert_raises

from flik import newton_solve


__all__ = [
    "test_newton_dimensions",
    "test_newton_inputs",
    "test_newton_linear_solve",
    "test_newton_no_solution1",
    "test_newton_nonlinear_solve",
    "test_newton_singular_matrix_error",
    ]


def f1(_):

    return np.zeros((2, 1))


def j1(_):

    return np.array([[1, 1], [-1, 1]])


def f2(_):

    return np.zeros((2,))


def j2(_):

    return np.zeros((2,))


def j3(_):

    return np.array([[1, 1], [-1, 1], [2, 3]])


def f4(x):

    return np.array([x[0] + x[1] - 3, x[1] - x[0] + 1])


def f5(x):

    return np.array([np.power(x[0], 3) + x[1] - 1, np.power(x[1], 3) - x[0] + 1])


def j5(x):

    return np.array([[3 * x[0] ** 2, 1], [-1, 3 * x[1] ** 2]])


def f6(x):

    return np.array([x[0] ** 2 + x[1] ** 2 - 1, -x[0] ** 2 + x[1] + 10])


def j6(x):

    return np.array([[2 * x[0], 2 * x[1]], [-2 * x[0], 1]])


def f7(x):

    return np.array([-2. * x[0] ** 2. - (4. / 3.) * x[1] ** 3.,
                     -2. * x[0] ** 2. - (4./3.) * x[1] ** 3.])


def j7(x):

    return np.array([[4. * x[0], 4. * x[1] ** 2.],
                     [4. * x[0], 4 * x[1]**2.]])


def test_newton_inputs():
    """Test invalid inputs to newton_solve."""
    # Initial guess
    x_0 = np.array([1.0, 1.0])

    # Check validity of inputs
    assert_raises(TypeError, newton_solve, "string", j1, x_0)
    assert_raises(TypeError, newton_solve, f1, "string", x_0)
    assert_raises(TypeError, newton_solve, f1, j1, "string")
    assert_raises(TypeError, newton_solve, f1, j1, x_0, "string")
    assert_raises(ValueError, newton_solve, f1, j1, x_0, -1.0)
    assert_raises(ValueError, newton_solve, f1, j1, x_0, eps=-1)
    assert_raises(TypeError, newton_solve, f1, j1, x_0, maxiter=0.)
    assert_raises(ValueError, newton_solve, f1, j1, x_0, maxiter=-1)


def test_newton_dimensions():
    """Test invalid function output dimensions in newton."""
    # Initial guess
    x_0 = np.array([1.0, 1.0])
    # Check valid dimensions of function
    assert_raises(TypeError, newton_solve, f1, j1, x_0)
    # Check valid dimensions of Jacobian
    assert_raises(TypeError, newton_solve, f2, j2, x_0)
    assert_raises(TypeError, newton_solve, f2, j3, x_0)


def test_newton_linear_solve():
    """Test that newton solves linear systems in 1 step."""
    x_0 = np.array([1.0, 1.0])
    result = newton_solve(f4, j1, x_0, eps=1.0e-9, maxiter=1)
    assert result["success"]
    assert result["message"] == "Convergence obtained."
    assert result["niter"] == 1
    assert np.allclose(result["f"], [0., 0.], atol=1.0e-6)
    assert np.allclose(result["x"], [2., 1.], atol=1.0e-9)
    assert np.allclose(result["f"], [0., 0.], atol=1.0e-6)
    assert np.allclose(result["J"], [[1., 1.], [-1., 1]], atol=1.0e-6)


def test_newton_nonlinear_solve():
    """Test that newton solves nonlinear systems."""
    x_0 = np.array([0.5, 0.5])
    result = newton_solve(f5, j5, x_0, eps=1.0e-9, maxiter=100)
    assert result["success"]
    assert result["niter"] < 101
    assert np.allclose(result["f"], [0., 0.], atol=1.0e-6)
    assert result["message"] == "Convergence obtained."
    assert result["niter"] < 100
    assert np.allclose(result["x"], [1., 0.], atol=1.0e-6)
    assert np.allclose(result["f"], [0., 0.], atol=1.0e-6)
    assert np.allclose(result["J"], [[3., 1.], [-1., 0]], atol=1.0e-6)


def test_newton_no_solution1():
    """Test that no solution in newton_solve raises error."""
    x_0 = np.array([1., 1.])
    result = newton_solve(f6, j6, x_0, eps=1.0e-9, maxiter=100)
    assert not result["success"]
    assert not np.allclose(result["f"], [0., 0.], atol=1.0e-3)
    assert result["message"] == "Maximum number of iterations reached."
    assert result["niter"] == 100


def test_newton_singular_matrix_error():
    """Test that newton raises error when Jaocbian is singular."""
    x0 = np.array([5., 5.])

    result = newton_solve(f7, j7, x0)
    message_expt = "Singular Jacobian; no solution found."
    assert not result['success']
    assert result["message"] == message_expt
    assert result["eps"] == 1e-6
