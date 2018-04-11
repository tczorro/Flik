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

Gauss–Newton Algorithm
======================

The advantage of Gauss-Newton Method is that the second derivatives are not necessarily needed to be
computed.The limitation is also obvious since it can only minimize the sum of squared function values. The Gauss-Newton is mainly used to resolve the non-linear least squares
problems. It is named after two famous mathematicians Carl Friedrich Gauss and Isaac Newton.

The Gauss-Newton Method for least squares problems can be interpreted as

.. math::
    (J^{\top} J)(x_{s+1} - x_s) = -J^{\top} f

where :math:`J` is the Jacobian matrix, :math:`J^{\top}` is the transpose of Jacobian, :math:`s` and
:math:`s+1` are the indexes. Now the problem becomes

.. math::
    x_{s+1} = x_s - {(J^{\top} J)}^{-1} {J^{\top}} f(x_s)

where :math:`x_0` is the initial guess. When the matrix :math:`f` is squared, the simplified
formula of the iteration evolves into the special case of Newton's Method

.. math::
    \begin{equation}
        x_{s+1} = x_s - J^{-1} f(x_s)
    \end{equation}
