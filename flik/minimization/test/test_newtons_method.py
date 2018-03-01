#!/usr/bin/env python3

# test for newtons_method.py
# March 1, 2018

from flik.minimization import newtons_method

class SingleVarFunction(object):

    def __init__(self, coefficients, order):
        self.coefficients = coefficients    # list
        self.order = order                  # integer
        if self.order != len(self.coefficients)-1:
            raise ValueError("Order must be equal to the length of the coefficient list minus one.")

    def eval_function(self, x):
        """Evaluate the function at the given point.

        Parameters
        ----------
        x : float
            The point at which to evaluate the function.

        Returns
        -------
        float
            The value of the function at the given point.

        """
        result = 0
        for i, c in enumerate(range(self.order, -1, -1)):
            result += self.coefficients[i]*x**c
        return result

def test_newtons_method():
    """ test newtons method :)
    """
    # quadratic 5x^2 - 2x + 3
    f2 = SingleVarFunction([5, -2, 3], 2)
    f2 = f2.eval_function
    # 10x - 2
    g2 = SingleVarFunction([10, -2], 1)
    g2 = g2.eval_function
    # 10
    h2 = SingleVarFunction([10], 0)
    h2 = h2.eval_function
    # initial point
    ip2 = 0.25
    res = newtons_method.newtons_opt(f2, g2, h2, ip2)

    # cubic -3x^3 + 10x^2 + 2x + 4
    f3 = SingleVarFunction([-3, 10, 2, 4], 3)
    f3 = f3.eval_function
    # -9x^2 + 20x + 2
    g3 = SingleVarFunction([-9, 20, 2], 2)
    g3 = g3.eval_function
    # -18x + 20
    h3 = SingleVarFunction([-18, 20], 1)
    h3 = h3.eval_function
    # initial point
    ip3 = 2.3

    # quartic x^4 + 2x^3 + 3x^2 + 4x + 5
    f4 = SingleVarFunction([1, 2, 3, 4, 5], 4)
    f4 = f4.eval_function
    # 4x^3 + 6x^2 + 6x + 4
    g4 = SingleVarFunction([4, 6, 6, 4], 3)
    g4 = g4.eval_function
    # 12x^2 + 12x + 6
    h4 = SingleVarFunction([12, 12, 6], 2)
    h4 = h4.eval_function
    # initial point
    ip4 = -1.05

    # quintic 2x^5 - x^4 + 3x^3 + 4x^2 + 3x + 2
    f5 = SingleVarFunction([2, -1, 3, 4, 3, 2], 5)
    f5 = f5.eval_function
    # 10x^4 - 4x^3 + 9x^2 + 8x + 3
    g5 = SingleVarFunction([10, -4, 9, 8, 3], 4)
    g5 = g5.eval_function
    # 40x^3 - 12x^2 + 18x + 8
    h5 = SingleVarFunction([40, -12, 18, 8], 3)
    h5 = h5.eval_function
    # initial point
    ip5 = -0.75

    # result
    print(res)
    return res

if __name__ == "__main__":
    test_newtons_method()

