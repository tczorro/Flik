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

Quasi–Newton Algorithm
======================

Quasi-Newton methods provide a way to approximate and update the derivative of a vector-valued
function during an iterative process. As Newton methods, they can be applied to find zeros
(approximating the Jacobian) or function minimization problems (approximating the Hessian). Using
the function and direction vectors from the previous step, quasi-Newton methods update the Jacobian
needed to solve a nonlinear system of equations.

Quasi-Newton methods are useful for situations where the actual derivative is unknown or
computationally expensive, being faster than Newton method for the latter case. However, the lack of
precision in the computed derivatives makes quasi-Newton algorithms take more steps to converge,
which becomes a disadvantage for simpler problems.
