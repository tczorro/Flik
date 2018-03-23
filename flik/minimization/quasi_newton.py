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


# API: method_name_B(B, secant, step): -> new_B
# or method_name_H(H, secant, step): -> new_H
# testfilename should be test_method_name.py in minimization/test

import numpy as np

def update_hessian_bfgs(hessian, pointk, pointkp1):
    """BFGs update for quasi-newton.
    
    Equation 6.19 from numerical optimisation book.
    """
    sk = pointkp1 - pointk
    yk = grad(pointkp1) - grad(pointk)
    newhess = hessian(pointk)
    # part one
    one = np.dot(np.dot(np.dot(sk, sk.T), hessian(pointk)), hessian(pointk))
    one /= np.dot(np.dot(sk.T,hessian(pointk)), sk)
    newhess -= one
    # part two
    newhess += np.dot(yk, yk.T)/np.dot(yk.T, sk)
    return newhess





























