安装指南
########


版本需求
========
* Python 3: HyperTS需要Python版本为3.7或者3.8。

* Tensorflow >=2.0.0, HyperTS的深度学习模式基于Tensorflow2实现。

-----------

pip安装
========

.. code-block:: sh

    pip install hyperts


conda安装
==========

.. code-block:: sh

    conda install -c conda-forge hyperts

-----------


注意事项
==========

1. HyperTS依赖prophet, 在使用``pip``安装hyperts时, 建议您使用 ``conda`` 先来安装prophet, 然后再安装hyperts:

.. code-block:: sh

    conda install -c conda-forge prophet==1.0.1
    pip install hyperts

2. Tensorflow对于HyperTS是可选依赖。当您深度学习及神经架构搜索模式时, 可手动安装tensorflow, 例如:

.. code-block:: sh

    conda install -c conda-forge prophet==1.0.1
    pip install hyperts tensorflow

或

.. code-block:: sh

    conda install -c conda-forge hyperts
    pip install tensorflow

3. 如果您的设备支持GPU, 您可以手动安装tensorflow-gpu版本来给深度学习模型加速。

4. 如果您使用hyperts时在深度学习模式下运行出现错误:

.. code-block:: none

    Frappe installation error "AttributeError: module 'pyparsing' has no attribute 'downcaseTokens'".

建议您控制pyparsing版本小于或等于2.4.7。

5. 在使用tensorflow时, 也许会遇到如下一些问题:
   
.. code-block:: none

    NotImplementedError: Cannot convert a symbolic Tensor (gru_1/strided_slice:0) to a numpy array. 
    This error may indicate that you're trying to pass a Tensor to a NumPy call, which is not supported.

您可以注意一下numpy版本和正在使用的tensorflow版本的兼容性, 适当降低numpy版本(如1.19.5), 也许会避免不必要的调试。
