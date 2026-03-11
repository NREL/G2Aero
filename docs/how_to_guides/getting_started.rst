.. _install:

Getting started
===============

The below will help you quickly install G2Aero.

Requirements
------------

You will need a working Python 3.9+ installation;
You will also need to install the following packages:

    * numpy
    * scipy
    * PyYAML

Installing via conda-forge
--------------------------

.. code-block:: bash
    
    conda install -c conda-forge g2aero


Install from source
-------------------

Alternatively, you can install the latest version directly from the most up-to-date version
of the source code by cloning/forking the GitHub repository

.. code-block:: bash

    git clone https://github.com/NREL/G2Aero.git


Once you have the source, you can build G2Aero (and add it to your environment)
by executing

.. code-block:: bash

    pip install .

in the top-level directory. The required Python packages will automatically be
installed as well.

.. note::

    **Legacy install (Python 3.6–3.8):** The ``setup.py`` file was removed in
    favor of ``pyproject.toml``. For older Python versions, check out tag
    ``v1.0.0`` or earlier (``git checkout v1.0.0``) and build with
    ``python setup.py install``.

You can test your installation by looking for the g2aero
executable built by the installation

.. code-block:: bash

    which g2aero

and by importing the g2aero Python frontend in Python

.. code-block:: python

    import g2aero

Testing
-------

To test that the package is working correctly, run

.. code-block:: bash

    pytest

from the root directory of the package.
This will run a basic test problem.