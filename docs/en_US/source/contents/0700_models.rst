模式模型
########

HyperTS在时间序列分析上平行地支持统计模型模式, 深度学习模式以及神经架构搜索模式(暂时未开放)。三种模式内置了多种优秀的模型, 例如Prophet, ARIMA, DeepAR, LSTNet等。在未来, 我们将继续丰富更多的模型, 例如Transformer, N-Beats等。

--------

统计模型
********
时序预测: Prophet | ARIMA | VAR

时序分类: TSForest | KNeighbors

--------

Prophet
=======
Prophet使用一个可分解的时间序列模型, 主要由趋势项(trend), 季节项(seasonality)和假期因素(holidays)组成:

.. math::
    y(t)=g(t)+s(t)+h(t)+\epsilon_{t}, 

这里, :math:`g(t)` 是趋势函数,代表非周期变化的值, :math:`s(t)` 表示周期性变化(如每周和每年的季节性), :math:`h(t)` 表示在可能不规律的时间表上发生的假期的影响。误差项 :math:`\epsilon_{t}` 代表模型不能适应的任何特殊变化,并假设其符合正态分布。

.. tip::

    适用范围: 单变量时序预测。

--------

ARIMA
=====
ARIMA全称为自回归集成移动平均模型(Autoregressive Integrated Moving Average Model), 也可以称之为差分自回归移动平均模型ARIMA(p, d, q)。其中, AR(p)是自回归项, 是指差分序列的滞后, MA(q)是移动平均项, 是指误差项的滞后,而I(d)是用于时间序列平稳的差分阶数。
通常情况下, p阶AR模型可表示为:

.. math::
    X_{t}=\alpha _{1}X_{t-1}+\alpha _{2}X_{t-2}+...+\alpha _{p}X_{t-p}+\epsilon _{t},

这里, :math:`\epsilon _{t}` 表示一个扰动的白噪声。自回归模型首先需要确定一个阶数 :math:`p`, 表示用几期的历史值来预测当前值。如果 :math:`\epsilon _{t}` 不是一个白噪声 :math:`u _{t}`, 通常认为它是一个q阶的移动平均,即:

.. math::
    u _{t}=\varepsilon _{t}+\beta _{1}\varepsilon _{t-1}+...+\beta _{q}\varepsilon _{t-q},

其中, :math:`\varepsilon _{t}` 表示白噪声序列。特别地,当 :math:`u _{t}` 时, 即时间序列当前值与历史值无关, 而只依赖于历史白噪声的线性组合, 则得到MA模型:

.. math::
    X _{t}=\varepsilon _{t}+\beta _{1}\varepsilon _{t-1}+...+\beta _{q}\varepsilon _{t-q}.

值得注意的是, AR模型中历史白噪声的影响是间接影响当前预测值的, 即通过历史时序值。

而将AR(p)模型和MA(q)模型结合, 便可以得到一个一般化的自回归移动平均模型ARMA(p, q):

.. math::
    X_{t}=\alpha _{1}X_{t-1}+\alpha _{2}X_{t-2}+...+\alpha _{p}X_{t-p}+\varepsilon _{t}+\beta _{1}\varepsilon _{t-1}+...+\beta _{q}\varepsilon _{t-q}.

如果原数据不稳定(即 :math:`d\neq 0`), 那么就做差分, 通过ADF检验直到时间序列平稳。最后, 便可以得到ARIMA模型。

.. tip::

    适用范围: 单变量时序预测。

--------

VAR
===
VAR全称为向量自回归模型(Vector Autoregressive), 针对于多变量时序分析。在这个模型中, 一组向量里的每一个时间序列被模型化为决定于自己滞后项以及这组向量里所有其他变量的滞后项。例如, 两阶的VAR模型可以表示为:

.. math::
    x_{t}=\alpha _{1}x_{t-1}+\alpha _{2}x_{t-2}+...+\alpha _{p}x_{t-p}+\epsilon _{1t},\\
    y_{t}=\alpha _{1}y_{t-1}+\alpha _{2}y_{t-2}+...+\alpha _{p}y_{t-p}+\epsilon _{2t}.

VAR模型与AR模型相同, 一个核心问题是找到滞后项的阶数 :math:`p`, 从而获得好的预测效果。

.. tip::

    适用范围: 多变量时序预测。

--------

TSForest
========
TSForest全称为时间序列森林(Time Series Forest), 是一种针对时间序列分类的集成树模型。时间序列森林将时间序列转化为子序列的均值、方差和协方差等统计特征,通过使用随机森林(以每个间隔的统计信息作为特征)克服间隔特征空间巨大的问题,利用熵增益和距离度量的组合,用于评估分割。

详情可参看: `A Time Series Forest for Classification and Feature Extraction <https://arxiv.org/pdf/1302.2277>`_

.. tip::

    适用范围: 单变量时序分类。

--------

KNeighbors
==========
KNeighbors是采用k近邻的方式对时间序列进行分类的方法。不同于传统的K近邻基于欧式距离进行度量,这里将采用 `动态时间弯曲距离 <https://en.wikipedia.org/wiki/Dynamic_time_warping>`_ (Dynamic Time Warping, DTW)作为一种新的相似性度量方法,通过调节时间点之间的对应关系,能够寻找两个任意长时间序列中数据之间的最佳匹配路径,对噪声有很强的鲁棒性,可以更有效地度量时间序列的相似性。由于DTW距离不要求两个时间序列中的点一一对应,因此具有更广的适用范围。除此之外,还可以采用微分动态时间弯曲距离(Derivative Dynamic Time Warping, DDTW), 加权动态时间弯曲距离(Weighted Dynamic Time Warping, WDTW)等变种或者 `最长公共子序列 <https://en.wikipedia.org/wiki/Longest_common_subsequence_problem>`_ (Longest Common Subsequence, LCSS)等时序距离度量。

适用范围: 单/多变量时序分类。

-----------

深度学习
********
DeepAR | HybirdRNN | LSTNet

--------

DeepAR
======
DeepAR是基于深度学习的时间序列预测算法, 为升级版的自回归模型。与传统主流的利用循环神经网络来做时序预测的方法不同, DeepAR并不是直接简单地输出一个确定的预测值做点估计, 而是输出预测值的一个概率分布。这样预测可以带来两点好处: 一方面很多过程本身就具有随机属性, 因此输出一个概率分布更加贴近本质, 预测精确; 另一方面可以评估出预测的不确定性和相关等风险。

详情可参看: `DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks <https://arxiv.org/abs/1704.04110>`_

.. tip::

    适用范围: 单变量时序预测。

--------

HybirdRNN
=========
HybirdRNN模型是指朴素循环神经网络(Recurrent Neural Network, RNN), 门控循环单元网络(Gated Recurrent Unit, GRU)以及长短记忆网络(Long Short-term Memory, LSTM)三种循环神经网络的集合。众所周知, 循环神经网络是一类以序列数据为输入在序列的演进方向上捕获时间特性的深度学习模型。循环神经网络具有记忆性且参数共享, 为了预防深度网络的梯度消失或梯度爆炸等问题, LSTM分别引入了遗忘门, 输入门和输出门等门控机制来学习更长的序列信息。GRU与LSTM类似, 不过将三门减少重置门和更新门两个门控, 使得每个循环单元可以自适应的捕捉不同时间刻度下的依赖。GRU更容易训练, 不过二者的效果不分伯仲。

更多区别可参考: `Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling <https://arxiv.org/abs/1412.3555>`_

.. tip::

    适用范围: 单/多变量时序预测, 分类, 回归。

--------

LSTNet
======
LSTNet全称为长短时序网络(Long-and Short-term Time-series network, LSTNet), 是一种专门为长期和短期混合模式的多变量时间序列预测任务设计的深度学习框架。特点为: 1、通过一维卷积CNN来捕获短期局部信息; 2、使用LSTM或者GRU从来自卷积层的特征捕获长期的宏观信息; 3、对于输入数据维度整理, 使用SLTM或者GRU捕获更长期的信息并充分利用序列的周期特性; 4、用全连接网络模拟AR自回归过程, 为预测添加线性成份, 同时使输出可以响应输入的尺度变化。

详情可参看: `Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks <https://arxiv.org/abs/1703.07015>`_

.. tip::
    适用范围: 单/多变量时序预测,回归。

--------

神经架构搜索
*************
...