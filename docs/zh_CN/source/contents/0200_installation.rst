安装指南
########


版本需求
========
Python 3: HyperTS需要Python版本为3.7或者3.8。
Tensorflow >=2.0.0, HyperTS的深度学习模式基于Tensorflow2实现。

-----------

pip安装
========

.. code-block:: sh

    pip install hyperts

-----------


注意事项
==========

1. HyperTS依赖prophet, 在使用pip安装hyperts时, 建议您使用 ``conda`` 先来安装prophet:

.. code-block:: sh

    conda install -c conda-forge prophet==1.0.1

2. HyperTS依赖tensorflow, 为了加速安装, 建议您使用镜像源, 例如:

.. code-block:: sh

    pip install hyperts -i https://pypi.tuna.tsinghua.edu.cn/simple

3. 如果您的设备支持GPU, 您可以手动安装tensorflow-gpu版本来给深度学习模型加速。

4. 如果您使用hyperts时在深度学习模式下运行出现错误:

.. code-block:: none

    Frappe installation error "AttributeError: module 'pyparsing' has no attribute 'downcaseTokens'".

建议您控制pyparsing版本在2.4.7。

5. 在使用tensorflow时, 也许会遇到一些问题, 您可以注意一下numpy版本和正在使用的tensorflow版本的兼容性, 适当降低numpy版本, 也许会避免不必要的调试。
