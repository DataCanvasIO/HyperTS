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


Notes
==========

1. HyperTS relies on Prophet. When installing hyperts using pip, it is recommended that you first install Prophet using ``conda``:

.. code-block:: sh

    conda install -c conda-forge prophet==1.0.1

2. If your device supports GPU, you can manually install the ``tensorFlow-gpu`` version to speed up the deep learning model.


1. If you meet the error shown below when using the DL mode, please check and change the ``pyparsing`` version to 2.4.7. 

.. code-block:: none

    Frappe installation error "AttributeError: module 'pyparsing' has no attribute 'downcaseTokens'".


3. If you meet other problems when using ``tensorflow``, please first check the compatibility of  ``numpy`` and ``tensorflow``. Sometimes, select lower ``numpy`` versions would help to solve the problems.
