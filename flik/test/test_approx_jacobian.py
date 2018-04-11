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


"""Test file for `flik.approx_jacobian`."""


import numpy as np
import numpy.testing as npt

from flik import ForwardDiffJacobian
from flik import CentralDiffJacobian


__all__ = [
    "test_finite_diff_jacobian_inputs",
    "test_forward_diff_jacobian_square",
    "test_central_diff_jacobian_square",
    "test_forward_diff_jacobian_rectangular",
    "test_central_diff_jacobian_rectangular",
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
    y = np.empty((3,), dtype=x.dtype)
    y[0] = x[0] * x[1]
    y[1] = x[0] ** 3 - np.sqrt(x[1])
    y[2] = 4.0 * x[0] * x[1] ** 2
    return y


def j2(x):
    """Test function."""
    y = np.empty((3, 2), dtype=x.dtype)
    y[0, 0] = x[1]
    y[0, 1] = x[0]
    y[1, 0] = 3.0 * x[0] ** 2
    y[1, 1] = -0.5 * x[1] ** (-0.5)
    y[2, 0] = 4.0 * x[1] ** 2
    y[2, 1] = 8.0 * x[0] * x[1]
    return y


# Run tests


def test_finite_diff_jacobian_inputs():
    """Test invalid inputs to FiniteDiffJacobian."""
    # Test `f` argument
    npt.assert_raises(TypeError, ForwardDiffJacobian, "string", 2)
    # Test `m` argument
    npt.assert_raises(TypeError, ForwardDiffJacobian, f1, 3.14159)
    # Test `n` argument
    npt.assert_raises(TypeError, ForwardDiffJacobian, f1, 2, 3.14159)
    # Test `eps` argument
    npt.assert_raises(TypeError, ForwardDiffJacobian, f1, 2, eps="string")
    # Test negative `n`
    npt.assert_raises(ValueError, ForwardDiffJacobian, f1, -2)
    # Test negative `m`
    npt.assert_raises(ValueError, ForwardDiffJacobian, f1, 2, -2)
    # Test `eps` array size
    npt.assert_raises(ValueError, ForwardDiffJacobian, f1, 2, eps=np.full((5,), 0.0078125))
    # Test negative `eps` scalar
    npt.assert_raises(ValueError, ForwardDiffJacobian, f1, 2, eps=-0.24681357)
    # Test negative `eps` array
    npt.assert_raises(ValueError, ForwardDiffJacobian, f1, 2, eps=np.full((2,), -0.0078125))


def test_forward_diff_jacobian_square():
    """Test ForwardDiffJacobian against square system `f1` and `J1`."""
    j = ForwardDiffJacobian(f1, 2)
    x = np.random.rand(2)
    jx = j(x)
    j1x = j1(x)
    diff = np.abs(jx - j1x)
    assert np.all(diff < 1.0e-3)

    j = ForwardDiffJacobian(f1, 2)
    x = np.random.rand(2)
    f1x = f1(x)
    jx = j(x, f1x)
    j1x = j1(x)
    diff = np.abs(jx - j1x)
    assert np.all(diff < 1.0e-3)

    j = ForwardDiffJacobian(f1, 2, eps=np.full((2,), 1.0e-3))
    x = np.random.rand(2)
    jx = j(x)
    j1x = j1(x)
    diff = np.abs(jx - j1x)
    assert np.all(diff < 5.0e-3)


def test_central_diff_jacobian_square():
    """Test CentralDiffJacobian against square system `f1` and `J1`."""
    j = CentralDiffJacobian(f1, 2)
    x = np.random.rand(2)
    jx = j(x)
    j1x = j1(x)
    diff = np.abs(jx - j1x)
    assert np.all(diff < 1.0e-3)

    j = CentralDiffJacobian(f1, 2, eps=np.full((2,), 1.0e-3))
    x = np.random.rand(2)
    jx = j(x)
    j1x = j1(x)
    diff = np.abs(jx - j1x)
    assert np.all(diff < 5.0e-3)


def test_forward_diff_jacobian_rectangular():
    """Test ForwardDiffJacobian against rectangular system `f2` and `J2`."""
    j = ForwardDiffJacobian(f2, 3, 2)
    x = np.random.rand(2)
    jx = j(x)
    j2x = j2(x)
    diff = np.abs(jx - j2x)
    assert np.all(diff < 1.0e-3)

    j = ForwardDiffJacobian(f2, 3, 2)
    x = np.random.rand(2)
    f2x = f2(x)
    jx = j(x, f2x)
    j2x = j2(x)
    diff = np.abs(jx - j2x)
    assert np.all(diff < 1.0e-3)

    j = ForwardDiffJacobian(f2, 3, 2, np.full((2,), 1.0e-4))
    x = np.random.rand(2)
    jx = j(x)
    j2x = j2(x)
    diff = np.abs(jx - j2x)
    assert np.all(diff < 5.0e-3)


def test_central_diff_jacobian_rectangular():
    """Test CentralDiffJacobian against rectangular system `f2` and `J2`."""
    j = CentralDiffJacobian(f2, 3, 2)
    x = np.random.rand(2)
    jx = j(x)
    j2x = j2(x)
    diff = np.abs(jx - j2x)
    assert np.all(diff < 1.0e-3)

    j = CentralDiffJacobian(f2, 3, 2, np.full((2,), 1.0e-3))
    x = np.random.rand(2)
    jx = j(x)
    j2x = j2(x)
    diff = np.abs(jx - j2x)
    assert np.all(diff < 5.0e-3)
