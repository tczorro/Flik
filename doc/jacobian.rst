..
    : An experimental local optimization package
    : Copyright (C) 2018 Ayers Lab <ayers@mcmaster.ca>.
    :
    : This file is part of Flik.
    :
    : Flik is free software; you can redistribute it and/or
    : modify it under the terms of the GNU General Public License
    : as published by the Free Software Foundation; either version 3
    : of the License, or (at your option) any later version.
    :
    : Flik is distributed in the hope that it will be useful,
    : but WITHOUT ANY WARRANTY; without even the implied warranty of
    : MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    : GNU General Public License for more details.
    :
    : You should have received a copy of the GNU General Public License
    : along with this program; if not, see <http://www.gnu.org/licenses/>

Jacobian Matrix
===============

Definition
----------

Suppose we have a vector valued function
:math:`f: \mathbb{R}^n \rightarrow \mathbb{R}^m`. This function will take the input vector
:math:`x \in \mathbb{R}^n` and map it to
the result vector :math:`f(x) \in \mathbb{R}^m`. The Jacobian is also called
:math:`Df, J_f, \frac{\partial (f_1, \cdots, f_m)}{ \partial (x_1, \cdots, x_n)}` The Jacobian matrix is defined by

.. math::
  J & = \left[
             \frac{\partial f}{\partial x_1} \frac{\partial f}{\partial x_2}
             \cdots \frac{\partial f}{\partial x_n}
         \right]
    & = {
    \begin{bmatrix}
      \frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\
      \vdots         & \ddots & \vdots \\
      \frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n}
    \end{bmatrix}
    }

Or element-wise,

.. math::
  J_{ij} = \frac{\partial f_i}{\partial x_j}

Example
------------

For example, if we have a function :math:`f: \mathbb{R}^2 \rightarrow \mathbb{R}^2`,

.. math::
  f(x,y) = { \begin{bmatrix} {x^2}y\\
               5x + siny
             \end{bmatrix}
           }

Then

.. math::
  f_1{ (x,y)} = {x^2}y \\
  f_2{ (x,y)} = 5x + \sin y

We can therefore determine the Jacobian matrix,

.. math::
  J_f(x,y) & = {\begin{bmatrix}
                  \frac{\partial f_1}{\partial x} & \frac{\partial f_1}{y}\\
                  \frac{\partial f_2}{x}          & \frac{\partial f_2}{y}
                \end{bmatrix}
               }
           & = {\begin{bmatrix}
                  2xy & x^2 \\
                  5   & \cos y
                \end{bmatrix}
               }

Solver
------

In our package, we have two different ways to approximate Jacobian, namely the forward difference
method and the central difference method. Both these two methods are based on the principle that the
derivatives in the partial differential equation can be approximated by the linear combinations of
function values. The central difference method is more accurate than the forward difference method,
but it requires 2 additional :math:`f(x)` evaluations while the forward difference method only needs
1 evaluation. The key iteration for forward difference method is defined by

.. math::
    J_{ij} = \frac{\partial f_i(x)}{\partial x_j}
           = \frac{f(x + \epsilon e_j) - f(x)} {\epsilon}

Then the iteration of central difference method can be interpreted

.. math::
    J_{ij} = \frac{\partial f_i(x)}{\partial x_j}
           = \frac{f(x + \epsilon e_j) - f(x - \epsilon e_j)} {2 \epsilon}

where the :math:`e_j` is the unit vector in dimension :math:`j` and :math:`\epsilon` is the small
finite increment.
