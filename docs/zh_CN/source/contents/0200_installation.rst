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

conda安装
==========

.. code-block:: sh

    conda install hyperts

-----------

注意事项
==========

1. 如果您使用时选择到统计模型模式, 安装sktime或者prophet显示失败, 建议您使用conda而非pip来安装:

.. code-block:: sh

    conda install -c conda-forge sktime==0.8.1

.. code-block:: sh

    conda install -c conda-forge prophet==1.0.1

2. 如果您使用时使用到深度学习模式, 并安装了tensorflow, 运行出现错误:

.. code-block:: none

    Frappe installation error "AttributeError: module 'pyparsing' has no attribute 'downcaseTokens'".

建议您控制pyparsing版本在2.4.7。

3. 在使用tensorflow时, 也许会遇到一些问题, 您可以注意一下numpy版本和正在使用的tensorflow版本的兼容性, 适当降低numpy版本, 也许会避免不必要的调试。
