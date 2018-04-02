"""MultiVarFunction class from flik.

This module contains a class MultiVarFunction that
inherits from the object class. The purpose of this object
is to provide callables that are evaluated for functions
at a given point.

Notes
-----
Need to come up with an option where we can have two
of the same coefficient for different variables.
i.e. x^2+y^2 is currently:
{1:[2,0], 1:[0,2]}
Small fix could be 1 and 1.0 but would like something better.

Example
-------
Create an object:
>>> my_func = MultiVarFunction({5: [2,3], 6: [2,1], -3: [0,0]}, 2)

This object can be printed nicely:
>>> print(my_func)
f(x, y) = 5x^2y^3 + 6x^2y - 3

You can construct a gradient for the object:
>>> my_func.construct_grad()
array([{10: [1, 3], 12: [1, 1]}, {15: [2, 2], 6: [2, 0]}], dtype=object)

You can construct a hessian for the object:
>>> my_func.construct_hess()
array([[{10: [0, 3], 12: [0, 1]}, {30: [1, 2], 12: [1, 0]}],
       [{30: [1, 2], 12: [1, 0]}, {30: [2, 1]}]], dtype=object)

.. _flik Documentation HOWTO:
    https://github.com/QuantumElephant/Flik

"""

import numpy as np


class MultiVarFunction(object):
    """Multiple variable function object.

    Parameters
    ----------
    structure : dict
        The keys are the values of the coefficients.
        The values of the keys is a list, where:
        structure[coeff1] = [x_order, y_order, z_order]
        Example: f(x,y) = axy^2 + bx^2 + cy + d
        structure[a] = [1, 2]
        structure[d] = [0, 0]
    num_var : int
        The number of variables in the function.

    Returns
    -------
    None

    """

    def __init__(self, structure, num_var):
        """Run constructor for MultiVarFunction class."""
        self.structure = structure
        self.num_var = num_var

    def __str__(self):
        """Overload the string representation of the function."""
        # define variables to use
        # cannot use vars (python keyword)
        svars = ['x', 'y', 'z']
        # don't include f
        svars += [chr(i) for i in range(97, 102)]
        svars += [chr(i) for i in range(103, 120)]
        # result string
        res = 'f('
        for i in range(self.num_var):
            res += svars[i]+', '
        res = res[:-2] + ') ='
        # put actual function here
        for i, coeff in enumerate(self.structure):
            if coeff >= 0 and i != 0:
                res += " + "+str(coeff)
            elif coeff < 0:
                res += " - "+str(abs(coeff))
            else:
                res += " "+str(coeff)
            for h in range(self.num_var):
                if self.structure[coeff][h] == 1:
                    res += svars[h]
                elif self.structure[coeff][h] > 1:
                    res += svars[h]+'^'
                    res += str(self.structure[coeff][h])
        return res

    def __call__(self, point):
        """Evaluate the function at the given point.

        Parameters
        ----------
        point : list / numpy.array
            The point at which to evaluate the function.
            The order of evaluation should be the same
            as for the dictionary structure.
            Example:
            f(x,y,z) = 10x^2y^3z^1
            point = [x,y,z]
            structure = { 10: [2,3,1] }

        Returns
        -------
        float
            The value of the function at the given point.

        Raises
        ------
        ValueError
            The length of the point list is not the same
            as the number of variables (num_var).

        """
        if len(point) != self.num_var:
            raise ValueError("Length of point list should be equal to the number of variables.")
        result = 0
        for coeff in self.structure:
            result += coeff*np.prod([point[i]**x for i, x in enumerate(self.structure[coeff])])
        return result

    def derivative(self, func, varnum):
        """Take the derivative of a function with n variables.

        Parameters
        ----------
        func : dictionary
            Specifies the function to be differentiated.
        varnum : int
            How many variables are in the function.

        Notes
        -----
        Take the derivative of the input function.
        Given variable number (index number for variable
        list) and function (as a dictionary).
        Returns dictionary (derivative function).

        Returns
        -------
        deriv : dictionary
            A dictionary representing the differentiated
            function.

        """
        deriv = {}
        for coeff in func:
            exp = func[coeff][varnum]
            # if not a constant (derivative disappears)
            if exp != 0:
                deriv[coeff*exp] = func[coeff][:]
                # decrement exponent
                deriv[coeff*exp][varnum] -= 1
        # construct empty derivative if necessary
        if deriv == {}:
            deriv = {0: [0]*self.num_var}
        return deriv

    def construct_grad(self):
        """Construct a gradient for the parent function."""
        # initialise empty gradient
        grad = np.zeros(shape=self.num_var, dtype=dict)
        # loop over number of variables
        for i in range(self.num_var):
            # take the derivative with respect to each
            # variable
            d = self.derivative(self.structure, i)
            grad[i] = MultiVarFunction(d, self.num_var)
        return Gradient(grad)

    def construct_hess(self):
        """Make a hessian, given the input function.

        Notes
        -----
        Assumes that the second derivatives
        of the function are all continuous (i.e.
        constructs a symmetric nxn matrix, where
        n is the number of variables in the function).

        """
        # initialise empty hessian
        hess = np.zeros(dtype=object,
                        shape=(self.num_var, self.num_var))
        # first get gradient
        grad = self.construct_grad()
        # define start index
        sind = 0
        for i in range(self.num_var):
            for j in range(sind, self.num_var):
                m = self.derivative(grad[i].structure, j)
                hess[i][j] = MultiVarFunction(m, self.num_var)
                if i != j:
                    hess[j][i] = hess[i][j]
            sind += 1
        return Hessian(hess)


class Gradient(MultiVarFunction):
    """1D array of multiple variable function objects.

    Parameters
    ----------
    grad : list / np.array
        A list of MultiVarFunction objects.

    Returns
    -------
    None

    """

    def __init__(self, grad):
        """Run constructor for Gradient class."""
        self.grad = grad

    def __len__(self):
        """Return the number of functions in the gradient."""
        return len(self.grad)

    def __str__(self):
        """Make a string for each function in the gradient."""
        result = ""
        for func in self.grad:
            result += str(func) + '\n'
        return result[:-1]

    def __getitem__(self, index):
        """Get the value of the gradient at the given index."""
        return self.grad[index]

    def __setitem__(self, index, value):
        """Set the value of the gradient at the given index."""
        self.grad[index] = value

    def __call__(self, point):
        """Evaluate the gradient at the given point.

        Parameters
        ----------
        point : list / numpy.array
            The point at which to evaluate the gradient.

        Returns
        -------
        np.array
            The value of the gradient at the given point.

        """
        res = np.zeros(len(point))
        for i in range(len(point)):
            res[i] = self.grad[i](point)
        return res


class Hessian(MultiVarFunction):
    """2D array of multiple variable function objects.

    Parameters
    ----------
    grad : np.array
        A matrix containing Gradient objects.

    Returns
    -------
    None

    """

    def __init__(self, hess):
        """Run constructor for Hessian class."""
        self.hess = hess

    def __str__(self):
        """Construct instances of function strings."""
        result = ""
        for row in self.hess:
            for func in row:
                result += str(func) + '\n'
        return result[:-1]

    def __getitem__(self, index):
        """Get a row of the hessian."""
        return self.hess[index]

    def __call__(self, point):
        """Evaluate the hessian at the given point.

        Parameters
        ----------
        point : list / numpy.array
            The point at which to evaluate the hessian.

        Returns
        -------
        np.array
            The value of the hessian at the given point.

        """
        res = np.zeros(shape=(len(point), len(point)))
        for i in range(len(point)):
            for j in range(len(point)):
                res[i][j] = self.hess[i][j](point)
        return res


if __name__ == "__main__":
    # example usage
    FUNCTION = MultiVarFunction({3: [2, 1], -7: [1, 1], -9: [0, 0]}, 2)
    GRADIENT = FUNCTION.construct_grad()
    HESSIAN = FUNCTION.construct_hess()
    print(GRADIENT)
    print(HESSIAN)
    print(GRADIENT([2, 3]))
    print(HESSIAN([2, 3]))
