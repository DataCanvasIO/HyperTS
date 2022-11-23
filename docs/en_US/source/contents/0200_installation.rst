Installation
#############


Software Environment
=====================
* Python 3.7 or 3.8.

* Tensorflow >=2.0.0, (Deep learning models require Tensorflow2).



Install with pip
====================

.. code-block:: sh

    pip install hyperts


Install with conda
====================

.. code-block:: sh

    conda install -c conda-forge hyperts


Notes
==========

1.1 Prophet is required by HyperTS (version < 0.2.0). When installing hyperts using ``pip``, it is recommended that you first install Prophet using ``conda``:

.. code-block:: sh

    conda install -c conda-forge prophet==1.0.1
    pip install hyperts

1.2 Since HyperTS version 0.2.0, hyperts relaxes prophet's version (compatible with prophet==1.1.1), so it is possible to install ``hyperts`` without first installing ``prophet`` using ``conda``:

.. code-block:: sh

    pip install hyperts

2. Tensorflow is an optional dependency for HyperTS. You can install tensorFlow manually when deep learning and neural architecture search modes, for example:

.. code-block:: sh

    conda install -c conda-forge prophet==1.0.1
    pip install hyperts tensorflow

or

.. code-block:: sh

    conda install -c conda-forge hyperts
    pip install tensorflow

3. If your device supports GPU, you can manually install the ``tensorFlow-gpu`` version to speed up the deep learning model.


4. If you meet the error shown below when using the DL mode, please check and change the ``pyparsing`` version not more than 2.4.7. 

.. code-block:: none

    Frappe installation error "AttributeError: module 'pyparsing' has no attribute 'downcaseTokens'".


5. If you meet other problems when using ``tensorflow``, please first check the compatibility of  ``numpy`` and ``tensorflow``. Sometimes, select lower ``numpy`` versions (i.e, 1.19.5) would help to solve the problems.
   
.. code-block:: none

     NotImplementedError: Cannot convert a symbolic Tensor (gru_1/strided_slice:0) to a numpy array. 
     This error may indicate that you're trying to pass a Tensor to a NumPy call, which is not supported.

6. When using the STATS model for univariate forecasting, you may encounter the following problems:

.. code-block:: none

    ValueError: In models with integration (d > 0) or seasonal integration (D > 0)...

Please check the ``statsmodels`` version and control that it is not greater than 0.12.1.
