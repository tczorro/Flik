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


"""
An experimental local optimization package.

Copyright (C) 2018 Ayers Lab <ayers@mcmaster.ca>.

"""


from .nonlinear import newton_solve
from .nonlinear import gauss_newton_solve

from .approx_jacobian import ForwardDiffJacobian
from .approx_jacobian import CentralDiffJacobian
