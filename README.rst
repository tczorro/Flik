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


.. image:: https://img.shields.io/badge/python-3.6-blue.svg
    :target: https://docs.python.org/3.6/
.. image:: https://travis-ci.org/QuantumElephant/Flik.svg?branch=master
    :target: https://travis-ci.org/QuantumElephant/Flik
.. image:: https://codecov.io/gh/QuantumElephant/Flik/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/QuantumElephant/Flik
.. image:: https://anaconda.org/QuantumElephant/Flik/badges/version.svg
    :target: https://anaconda.org/QuantumElephant/Flik


Flik
####

*An experimental local optimization package.*


Installation
============

Flik can be installed with pip (system wide or in a virtual environment):

.. code:: bash

    pip install flik

Alternatively, you can install Flik in your home directory:

.. code:: bash

    pip install flik --user

Lastly, you can also install Flik with conda. (See
https://www.continuum.io/downloads)

.. code:: bash

    conda install -c QuantumElephant flik


Testing
=======

The tests can be executed as follows:

.. code:: bash

    nosetests flik


Background and usage
====================

*Put some more details here*

Release history
===============

- **2018-15-02** 0.0.0

  Initial Release


How to make a release (Github, PyPI and anaconda.org)
=====================================================

Before you do this, make sure everything is OK. The PyPI releases cannot be undone. If you
delete a file from PyPI (because of a mistake), you cannot upload the fixed file with the
same filename! See https://github.com/pypa/packaging-problems/issues/74

1. Update the release history.
2. Commit the final changes to master and push to github.
3. Wait for the CI tests to pass. Check if the README looks ok, etc. If needed, fix things
   and repeat step 2.
4. Make a git version tag: ``git tag <some_new_version>`` Follow the semantic versioning
   guidelines: http://semver.org
5. Push the tag to github: ``git push origin master --tags``
