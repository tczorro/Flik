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


"""Test file for `flik.jacobian`."""


import numpy as np
import numpy.testing as npt

from flik import Jacobian


__all__ = [
    "test_jacobian_inputs",
    "test_update_badbroyden_j1",
    "test_update_goodbroyden_j1",
    "test_update_dfp",
    "test_secant_condition_dfp",
    "test_positive_definiteness_dfp",
    "test_update_inv_bfgs_j1",
    "test_update_inv_bfgs_j3",
    "test_bfgs_secant_condition",
    ]


# Seed the numpy rng for consistency
np.random.seed(101010101)


# Define some analytical test functions and Jacobians


def f1(x):
    """Test function."""
    y = np.copy(x)
    y **= 2
    y[0] += x[1]
    y[1] -= x[0]
    return y


def j1(x):
    """Test function."""
    y = np.empty((2, 2), dtype=x.dtype)
    y[0, 0] = 2.0 * x[0]
    y[0, 1] = 1.0
    y[1, 0] = -1.0
    y[1, 1] = 2.0 * x[1]
    return y


def f2(x):
    """Test function."""
    return np.array([x[0]**2 + x[1]**2 - 1, 3. * x[0]**2. + x[0] * x[1]**2. + x[2],
                     x[2]**2. + x[1]])


def j2(x):
    """Test function."""
    return np.array([[2. * x[0], 2. * x[1], 0.],
                     [6. * x[0] + x[1]**2., 2. * x[0] * x[1], 1.],
                     [0., 1., 2. * x[2]]])


def test_jacobian_inputs():
    """Test invalid inputs to Jacobian class."""
    # Test adding a non-callable jacobian to the class.
    npt.assert_raises(TypeError, Jacobian, jac=5.)
    # Test Jacobian calls the correct function.
    jac_obj = Jacobian(jac=j1)
    x_0 = np.random.rand(2)
    assert np.allclose(jac_obj(x_0), j1(x_0))
    jac_obj_obj = Jacobian(jac_obj)
    assert np.allclose(jac_obj_obj(x_0), j1(x_0))


def test_update_goodbroyden_j1():
    """Test Jacobian.update_goodbroyden using analytical Jacobian and function."""
    # Initial point x_k
    x_0 = np.random.rand(2)
    # Function at initial x_k
    f1_0 = f1(x_0)
    # Jacobian at initial x_k
    j1_0 = j1(x_0)
    # Define arbitrary step
    dx = np.asarray([0.01, 0.01])
    # Take step dx
    x_1 = x_0 + dx
    # Function at x_{k+1}
    f1_1 = f1(x_1)
    # Compute analytical Jacobian at x_{k+1}`
    j1_1 = j1(x_1)
    delta_f = f1_1 - f1_0
    expected_ans = j1_0 + \
        np.outer((delta_f - j1_0.dot(dx)) / np.dot(dx, dx), dx)
    # Approximate Jacobian at (x_0 + dx) with Good Broyden method
    # from Jacobian and vector function at initial x_0
    Jacobian.update_goodbroyden(j1_0, x_1, x_1 - x_0, f1_1 - f1_0)
    assert np.allclose(j1_0, expected_ans)
    assert np.allclose(j1_1, expected_ans, atol=1e-1)


def test_update_badbroyden_j1():
    """Test Jacobian.update_badbroyden against analytical Jacobian and function."""
    # Initial point x_0
    x_0 = np.random.rand(2)
    # Function at initial x_k
    f1_0 = f1(x_0)
    # Inverse Jacobian at initial x_0
    j1_inv_0 = np.linalg.inv(j1(x_0))
    # Define arbitrary step and take step
    dx = np.asarray([0.01, 0.01])
    x_1 = x_0 + dx
    # Function at x_{k+1}
    f1_1 = f1(x_1)
    delta_f = f1_1 - f1_0
    # Compute analytical inverse Jacobian at `x_0 + dx`
    expected_ans = j1_inv_0 + np.outer((dx - j1_inv_0.dot(delta_f)) /
                                       np.dot(delta_f, delta_f), delta_f)
    # Approximate inverse Jacobian at (x_0 + dx) with Bad Broyden method
    # from inverse Jacobian and vector function at initial x_0
    Jacobian.update_badbroyden(j1_inv_0, x_1, dx, delta_f)
    assert np.allclose(j1_inv_0, expected_ans, atol=1e-1)
    # assert that the approximation jacobian matches the analytic jacobian.
    j1_1 = np.linalg.inv(j1(x_1))
    assert np.allclose(j1_1, expected_ans, atol=1e-1)


def test_update_dfp():
    """Test the definition of DFP."""
    # Test wikipedia definition/format.
    random_vecs = np.random.rand(10, 3) * 50. + 0.5
    x = random_vecs[0]
    dx = np.array([0.0001, 0.0001, 0.0001])
    df = f2(x + dx) - f2(x)
    b0 = j2(x)
    gamma = 1. / np.dot(df, dx)
    expected_answer = (np.eye(3)
                     - gamma * np.outer(df, dx)).dot(b0).dot(np.eye(3)
                                                             - gamma * np.outer(dx, df))
    expected_answer += gamma * np.outer(df, df)
    Jacobian.update_dfp(b0, x + dx, dx, df)
    assert np.allclose(b0, expected_answer)


def test_positive_definiteness_dfp():
    """Test that DFP is positive definite."""
    # It's crucial that the x values are greater than 0.5
    # or else positive definiteness is not required.
    random_vecs = np.random.rand(10, 3) * 50. + 0.5
    dx = np.array([0.5, 0.5, 0.5])
    x = random_vecs[0]
    dfp = np.eye(3)
    Jacobian.update_dfp(dfp, x + dx, dx, f2(x + dx) - f2(x))
    assert np.all(np.asarray([np.dot(x, dfp.dot(x)) for x in random_vecs]) > 0.)


def test_secant_condition_dfp():
    """Test that dfp update satisfies secant condition."""
    x = (np.random.rand(1, 3) * 50. - 10.)[0]
    dx = np.array([0.5, 0.5, 0.5])
    df = f2(x + dx) - f2(x)
    dfp = np.eye(3)
    Jacobian.update_dfp(dfp, x + dx, dx, f2(x + dx) - f2(x))
    assert np.allclose(df, dfp.dot(dx))


def test_update_sr1_j1():
    """Test Jacobian.update_sr1 using analytical Jacobian and function."""
    # Initial point x_k
    x_0 = np.random.rand(2)
    # Function at initial x_k
    f1_0 = f1(x_0)
    # Jacobian at initial x_k
    j1_0 = j1(x_0)
    # Define arbitrary step
    dx = np.asarray([0.01, 0.01])
    # Take step dx
    x_1 = x_0 + dx
    # Function at x_{k+1}
    f1_1 = f1(x_1)
    # Compute analytical Jacobian at x_{k+1}`
    j1_1 = j1(x_1)
    delta_f = f1_1 - f1_0
    tmp = delta_f - j1_0.dot(dx)
    expected_ans = j1_0 + (np.outer(tmp, tmp.T)) / np.dot(tmp.T, dx)
    # Approximate Jacobian at (x_0 + dx) with Good Broyden method
    # from Jacobian and vector function at initial x_0
    Jacobian.update_sr1(j1_0, x_1, x_1 - x_0, f1_1 - f1_0)
    assert np.allclose(j1_0, expected_ans)
    assert np.allclose(j1_1, expected_ans, atol=1e-1)


def test_update_sr1_inv_j1():
    """Test Jacobian.update_sr1_inv against analytical Jacobian and function."""
    # Initial point x_0
    x_0 = np.random.rand(2)
    # Function at initial x_k
    f1_0 = f1(x_0)
    # Inverse Jacobian at initial x_0
    j1_inv_0 = np.linalg.inv(j1(x_0))
    # Define arbitrary step and take step
    dx = np.asarray([0.01, 0.01])
    x_1 = x_0 + dx
    # Function at x_{k+1}
    f1_1 = f1(x_1)
    delta_f = f1_1 - f1_0
    # Compute analytical inverse Jacobian at `x_0 + dx`
    tmp = dx - j1_inv_0.dot(delta_f)
    expected_ans = j1_inv_0 + (np.outer(tmp, tmp.T)) / np.dot(tmp.T, dx)
    # Approximate inverse Jacobian at (x_0 + dx) with Bad Broyden method
    # from inverse Jacobian and vector function at initial x_0
    Jacobian.update_sr1inv(j1_inv_0, x_1, x_1 - x_0, f1_1 - f1_0)
    assert np.allclose(j1_inv_0, expected_ans, atol=1e-1)
    # assert that the approximation jacobian matches the analytic jacobian.
    j1_1 = np.linalg.inv(j1(x_1))
    assert np.allclose(j1_1, expected_ans, atol=1e-1)


def test_bfgs_secant_condition():
    """Test the secant condition for BFGS."""
    x = (np.random.rand(1, 3) * 50. - 10.)[0]
    dx = np.array([0.5, 0.5, 0.5])
    df = f2(x + dx) - f2(x)
    # Compute the updated bfgs Jacobian
    bfgs = np.eye(3)
    Jacobian.update_bfgs(bfgs, x + dx, dx, f2(x + dx) - f2(x))
    bfgs = np.linalg.inv(bfgs)
    assert np.allclose(df, bfgs.dot(dx), atol=1e-1)
