Installation
#############


Software Environment
=====================
* Python 3.7 or 3.8

* Tensorflow >=2.0.0, (Deep learning models require Tensorflow2)



Install with pip
====================

.. code-block:: sh

    pip install hyperts



Install with conda
=======================

.. code-block:: sh

    conda install hyperts




Notes
==========

1. If you are failed to install ``sktime`` or ``prophet`` when using the STATS mode, suggest to install with ``conda`` instead of ``pip``:

.. code-block:: sh

    conda install -c conda-forge sktime==0.8.1

.. code-block:: sh

    conda install -c conda-forge prophet==1.0.1

2. If you meet the error shown below when using the DL mode, please check and change the ``pyparsing`` version to 2.4.7. 

.. code-block:: none

    Frappe installation error "AttributeError: module 'pyparsing' has no attribute 'downcaseTokens'".


3. If you meet other problems when using ``tensorflow``, please first check the compatibility of  ``numpy`` and ``tensorflow``. Sometimes, select lower ``numpy`` versions would help to solve the problems.
