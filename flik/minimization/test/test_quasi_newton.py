"""Testing for newtons_method file from flik.

This module contains one function that tests newtons_method
for a single variable function, 5x^2 - 2x + 3.
It asserts that a single-variable function converges in one
step.

Example
-------
Run with nosetests or run from the terminal.

    $ nosetests3 flik
    $ python test_newtons_method.py


.. _flik Documentation HOWTO:
    https://github.com/QuantumElephant/Flik

"""

import numpy as np
from flik.minimization import quasi_newton
from flik.MultiVarFunction import MultiVarFunction, Gradient, Hessian
from numpy.testing import assert_almost_equal, assert_equal


def test_quasi_newton_quad1():
    """Test newtons_method for a single variable quadratic.

    Notes
    -----
    The test function is: 5x^2 - 2x + 3
    The test gradient is: 10x - 2
    The test hessian is: 10
    The test point is: [0.25]
    Expect result of newtons_method to be:
    (2.8, [0.2], 0.0, 0)

    Returns
    -------
    None

    """
    quad = MultiVarFunction({5: [2], -2: [1], 3: [0]}, 1)
    grad = MultiVarFunction({10: [1], -2: [0]}, 1)
    hess = MultiVarFunction({10: [0]}, 1)
    val = np.array([0.25])

    # check results function
    def success(res):
        assert_almost_equal(res[0], 2.8, decimal=4)
        assert_almost_equal(res[1], np.array([0.2]), decimal=4)
        assert_almost_equal(res[2], np.array([0.0]), decimal=4)

    # basically newton's method
    res = quasi_newton.quasi_newtons_opt(quad, grad, hess, val)
    success(res)
    # Broyden quasi-newton
    res = quasi_newton.quasi_newtons_opt(quad, grad, hess, val, quasi_newton.update_hessian_broyden)
    success(res)


def test_quasi_newton_quad2():
    """Test newtons_method for a 2 variable quadratic.

    Notes
    -----
    The test function is: 3x^2y - 7yx - 9
    The test gradient [df/dx, df/dy] is:
        [6xy-7y, 3x^2-7x]
    The test hessian ([d2f/dx2,  d2f/dxdy],
                      [d2f/dydx,  d2f/dy2]) is:
        [6y,  6x-7],
        [6x-7,   0]
    The test point is: [-6,0]
    Expect result of newtons_method to be:
    ?

    Returns
    -------
    None

    """
    # doesn't work yet
    jen = MultiVarFunction({3: [2, 1], -7: [1, 1], -9: [0, 0]}, 2)
    g = jen.construct_grad()
    h = jen.construct_hess()
    print(h([-6,0]))

    val = np.array([-6, 0])
    res = quasi_newton.quasi_newtons_opt(jen, g, h, val)

def test_update_hessian_bfgs():
    """Test the update_hessian_bfgs function.

    Makes sure that the gradient and hessians are
    evaluated properly by MultiVarFunction.

    Makes sure that the function returns a valid
    hessian of the correct value and shape.

    """
    # make the test function
    # 2xy^2 + 3x^2 + 7
    func = MultiVarFunction({2: [1,2], 3:[2,0], 7:[0,0]}, 2)
    # make the gradient
    grad = func.construct_grad()
    # check df/dx (partial wrt x)
    assert grad[0].structure == {2: [0,2], 6: [1,0]}
    # check df/dy (partial wrt y)
    assert grad[1].structure == {4: [1,1]}
    # make the hessian
    hess = func.construct_hess()
    # check d2f/dx2
    assert hess[0][0].structure == {6: [0,0]}
    # check d2f/dxdy
    assert hess[0][1].structure == {4: [0,1]}
    # check d2f/dydx
    assert hess[1][0].structure == {4: [0,1]}
    # check d2f/dy2
    assert hess[1][1].structure == {4: [1,0]}
    # evaluate hessian at arbitrary point
    hess_eval = hess([1,2])
    # check evaluation of hessian at point (1,2)
    assert np.equal(hess_eval, np.array([[6., 8.], [8., 4.]])).all()
    # choose some arbitrary points
    pk = np.array([2,3])
    pk1 = np.array([3,4])
    result = quasi_newton.update_hessian_bfgs(hess_eval, 
                                              grad, pk, pk1)
    # check the resulting array
    assert np.equal(result, np.array([
                    [-190+400/-44, -160+480/-44],
                    [-160+480/-44, -140+576/-44]])).all()


if __name__ == "__main__":
    test_quasi_newton_quad1()
    #test_quasi_newton_quad2()
    test_update_hessian_bfgs()
