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

Newton Algorithm
================

Newton's method (also knows as the Newton-Raphson method, Newton's iteration) is an algorithm that utilizes the first terms the taylor series of function :math:`f(x)` to find the roots of
:math:`f(x)=0`. For the Taylor series of :math:`f(x)` about the point
:math:`x_{s+1} = x_s + \epsilon` where :math:`s` and :math:`s+1` are indexes,

.. math::
    f(x_{s+1}) = f(x_s) + f^{\prime}(x_s) \epsilon + \frac{1}{2} f^{\prime \prime}(x_s) + \cdots

If we only keep the first term, we will get

.. math::
    f(x_{s+1}) \approx f(x_s) + f^{\prime}({x_s}) \epsilon

Because we are solving :math:`f(x)=0`, then for Newton's method, the problem can be formulated as

.. math::
    0 \approx f(x_s) + J (x_{s+1} - x_{s})
    J (x_{s+1} - x_{s}) = -f

Therefore, the root can be obtained by calculating this term iteratively,

.. math::
    x_{s+1} = x_s - \frac{f}{J}

where :math:`J` is the Jacobian matrix of :math:`f`.  Once the :math:`|x_{s+1} - x_s| \leq \epsilon` (the desired accuracy value), we will let the :math:`x_{s+1}` serve as the approximated root.
