#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# An experimental local optimization package
# Copyright (C) 2018 Ayers Lab <ayers@mcmaster.ca>
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


"""Flik setup script.

If you are not familiar with setup.py, just use pip instead:

    pip install Flik --user --upgrade

Alternatively, you can install from source with

    ./setup.py install --user
"""


from __future__ import print_function

import sys
from warnings import warn

from setuptools import setup


def get_version():
    """Load the version from version.py, without importing it.

    This function assumes that the last line in the file contains a variable
    defining the version string with single quotes.

    """
    with open('flik/version.py', 'r') as f:
        return f.read().split('=')[-1].replace('\'', '').strip()


def readme():
    """Load README.rst for display on PyPI."""
    with open('README.rst') as f:
        return f.read()


if __name__ == '__main__':

    if sys.version_info.major == 2:
        warn('Python 2 is being used; Flik is only built for Python 3.')

    setup(
        name='flik',
        version=get_version(),
        description='An experimental local optimization package',
        long_description=readme(),
        author='Ayers Lab',
        author_email='ayers@mcmaster.ca',
        url='https://github.com/QuantumElephant/Flik',
        packages=['flik', 'flik.test'],
        zip_safe=False,
        )
