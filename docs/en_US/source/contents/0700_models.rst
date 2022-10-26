Model References
#################

HyperTS provides three different methods to perform time series analysis, which are statistical methods, deep learning algorithms and neural architecture search algorithms(not implemented yet). Each method also includes several algorithms to solve specific problems. This section will give a breif introduction to these algorithms. Besides, more novel and advanced algorithms will be involved in the near future, like Transformer and N-Beats.  

---------

Statistical Methods
********************
Different tasks require different statistical methods, which are introduced in sequence in this subsection.

- Time series forecasting: Prophet | ARIMA | VAR
- Time series classification: TSForest | KNeighbors
- Time series anomaly detection: TSIsolationForest | TSOneClassSVM


Prophet
========
Prophet is a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects. It works best with time series that have strong seasonal effects and several seasons of historical data. Prophet is robust to missing data and shifts in the trend, and typically handles outliers well. 

Prophet is stated as a decomposable model with three main components: trend, seasonality, and holidays. 

.. math::
    y(t)=g(t)+s(t)+h(t)+\epsilon_{t}, 

Where :math:`g(t)` is the trend function which models non-periodic changes in the value of thetime  series, :math:`s(t)` represents  periodic  changes  (e.g.,  weekly  and  yearly  seasonality), :math:`h(t)` represents the effects of holidays which occur on potentially irregular schedules overone or more days. The error term :math:`\epsilon_{t}` represents any idiosyncratic changes which are not accommodated  by  the  model;  later  we  will  make  the  parametric  assumption  that the error is normally distributed.

For more information, please check the `website <https://facebook.github.io/prophet/>`_ and the paper `Forecasting at scale <https://peerj.com/preprints/3190/>`_.

.. tip::

    Prophet is applied to univariate times series forecasting.

---------

ARIMA
=====
Autoregressive Integrated Moving Average (ARIMA) model, is a forecasting algorithm that is used to predict the future values of time series based on its own past values. An ARIMA model is characterized by 3 terms: p, d, q

where,

- p is the order of the AR term, referring to the number of lags,
- q is the order of the MA term, referring to the number of lagged forecast errors,
- d is the number of differencing required to make the time series stationary

The AR part of ARIMA indicates that the time series is regressed on its own past data. The function of AR(p) is :

.. math::
    X_{t}=\alpha _{1}X_{t-1}+\alpha _{2}X_{t-2}+...+\alpha _{p}X_{t-p}+\epsilon _{t},

where, :math:`X_{t}` is the current value. :math:`\alpha_{}`is the coefficient. :math:`p` is the order. :math:`\epsilon _{t}` is the forecast error, which is considered as white noise.

The MA part of ARIMA indicates that the forecast error is a linear combination of past respective errors. The function of MA(q) is: 

.. math::
    X_{t}=\varepsilon _{t}+\beta _{1}\varepsilon _{t-1}+...+\beta _{q}\varepsilon _{t-q},

where, :math:`\varepsilon _{}` are the errors of the AR models of the respective lags. :math:`\beta_{}` is the coefficient. From the equation, we could see that the past errors impact the current value indirectly. 

The ARMA(p, q) model is combined with AR and MA models:

.. math::
    X_{t}=\alpha _{1}X_{t-1}+\alpha _{2}X_{t-2}+...+\alpha _{p}X_{t-p}+\varepsilon _{t}+\beta _{1}\varepsilon _{t-1}+...+\beta _{q}\varepsilon _{t-q}.

The I part of ARIMA is to perform difference operation to make the time series stationary. Normally, the order of d is zero or one.

.. tip::
   
    ARIMA is applied to univariate times series forecasting.

-------

VAR
===
The Vector Autoregressive(VAR) model is a multivariate time series model that relates current observations of a variable with past observations of both itself and other variables in the system. That means, the VAR model requires at least two time series, which also influence each other. Each time series is modeled as a function of the past values, which is the same as the AR(p) model: 

.. math::
    x_{t}=\alpha _{1}x_{t-1}+\alpha _{2}x_{t-2}+...+\alpha _{p}x_{t-p}+\epsilon _{t}

A second order VAR(2) model for two variables can be fomulated as below:

.. math::
   x_{1,t}=\alpha _{11,1}x_{1,t-1}+\alpha _{12,1}x_{2,t-1}+\alpha _{11,2}x_{1,t-2}+\alpha _{12,2}x_{2,t-2}+\epsilon _{1,t}, \\
   x_{2,t}=\alpha _{21,1}x_{1,t-1}+\alpha _{22,1}x_{2,t-1}+\alpha _{21,2}x_{1,t-2}+\alpha _{22,2}x_{2,t-2}+\epsilon _{2,t}

.. tip::
    
    VAR is applied to multivariate times series forecasting.

--------

TSForest
========
TSForest is short for Time Series Forest. It's a tree-ensemble method proposed for time series classification. TSForest employs a combination of entropy gain
and a distance measure, referred to as the Entrance (entropy and distance) gain, for evaluating the splits. In detail, it randomly samples features at each
tree node and has computational complexity linear in the length of time series, and can be built using parallel computing techniques. The temporal
importance curve is proposed to capture the temporal characteristics useful for classification. 

For more information, please refer to the paper `A Time Series Forest for Classification and Feature Extraction <https://arxiv.org/pdf/1302.2277>`_

.. tip::

    TSForest is applied to univariate times series classification.

--------

KNeighbors
==========
K-nearest-neighbor(KNN) classifiers with dynamic time warping `(DTW) <https://en.wikipedia.org/wiki/Dynamic_time_warping>`_ has been widely used for similarity measurement in time series classification, which is usually outperform kNN with Euclidean distance. DTW is robust to the distortion of the time axis and random noise. It allows non-linear alignments between two time series to accommodate sequences that are similar, but locally out of phase. Besides, it could adopt Derivative Dynamic Time Warping (DDTW), Weighted Dynamic Time Warping (WDTW) or `Longest Common Subsequence (LCSS) <https://en.wikipedia.org/wiki/Longest_common_subsequence_problem>`_ methods for distance measurement to further improve the performance.

.. tip::
    
    KNeighbour is applied to both univariate and multivariate times series classification.

---------

TSIsolationForest
===================
Isolation forest detects anomalies using isolation (how far data point is to the rest of the data), rather than modeling the normal points. In 2007, it was initialy developed by Fei Tony Liu as one of the original ideas in his PhD study .The significance of this research lies in its deviation from the mainstream philosophy underpinning most existing anomaly detctors at the time, where all the normal instances are profiled before anomalies are identified as instances that do not confrom to the distribution of the normal instances. Isolation forest introduces a different method that explicitly isolates anomalies using binary trees, demostrating a new prossibility of a faster anomaly detector that directly targets anomalies without profilling all the normal instances. The algorithm has a linear time complexity with a low constant and a low memory requirement, which works well with high volume data.

For more information, please check the `wikipedia <https://en.wikipedia.org/wiki/Isolation_forest>`_ and the paper `Isolation Forest <https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf?q=isolation-forest>`_.

.. tip::
    
    TSIsolationForest is applied to both univariate and multivariate times series anomaly detection.

---------

TSOneClassSVM
===============
One-Class SVM is an unsupervised learning technique to learn the ability to differentiate the test samples of a particular calss from other classes. One-SVM is one of the most convenient methods to approach One-Class Classification problem statements including anomaly detection. One-SVM works on the basic idea of minimizing the hypersphere of the single class of examples in training data and considers all the other samples outside the hypersphere to be outliers or out of training data distribution. 

.. tip::
    
    TSOneClassSVM is applied to both univariate and multivariate times series anomaly detection.

-----------


Deep Learning Algorithms
*************************


DeepAR | HybirdRNN | LSTNet | InceptionTime | N-Beats | VAE

------------

DeepAR
======
DeepAR is a methodology for producing accurate probabilistic forecasts, based on training an auto-regressive recurrent network model(RNN) on time series. Differring from the conventional RNN model, DeepAR outputs probabilistic forecasts instead of point value forecasts. On one hand, this provides a better forecast accuracy since most process are random. On the other hand, it could indicate the uncertainty and risks of the output to enable optimal decision making.  

For more information, please refer to the paper `DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks <https://arxiv.org/abs/1704.04110>`_

.. tip::
    
    DeepAR is applied to univariate times series forecasting.

------------

HybirdRNN
=========
HybirdRNN model is a combination of Recurrent Neural Networks (RNN), Gated Recurrent Unit (GRU) and Long Short-term Memory (LSTM). RNN are a well-known class of neural networks that models sequential data or time series data. They could take the information from prior inputs (memory) to influence the current input and output. And they share parameters across each layer of the network.  LSTM were developed to deal with the vanishing gradients problems that tranditional RNNs can encountered.  A common LSTM unit is composed of a cell, an input gate, an output gate and a forget gate, which enable LSTM to learn longer sequencial information.  GRU is like a LSTM but with few parameters: a reset gate and a update gate. 

For more information, please refer to the paper `Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling <https://arxiv.org/abs/1412.3555>`_

.. tip::
    HybirdRNN is applied to all tasks: uni/multi-variate forecasting, classification and regression.

-----------

LSTNet
========
LSTNet is short for Long-and Short-term Time-series network, which is a deep learning framework particularly designed for a mixture of long-term and short-term multivariate time series forecasting. In detail, LSTNet firstly uses the Convolution Neural Network (CNN) to extract short-term local dependency patterns among multi-dimensional variables. And it uses the Recurrent Neural Network (RNN) to discover long-term patterns for time series trends. Then LSTNet introduces a novel recurrent structure to capture very long-term dependence patterns and making the optimization easier as it utilizes the periodic property of the input time series signals. Lastly, it incorporates a traditional autoregressive model to tackle the scale insensitive problem of the neural network model. 

For more information, please refer to the paper `Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks <https://arxiv.org/abs/1703.07015>`_

.. tip::
    LSTNet is applied to uni/multi-variate forecasting and regression.

-----------

InceptionTime
===============
The network architecture of InceptionTime highly resembles to that of GoogleNet. In particular, the network consists of a series of Inception modules followed by a Global Average Pooing layer and a Dense layer with a softmax activation function. However, InceptionTime introduces an additional element within its network's layers: residual connections at every third inception module. For computer vision problems, we expect our model to learn features in a similar fashion. Similarly, InceptionTime expects the bottom-layer neurons to capture the local structure of a time series such as lines and curves, and the top-layer neurons to identify various shape patterns such as 'valleys' and 'hills'.

For more information, please refer to the paper `InceptionTime: Finding AlexNet for time series classification <https://link.springer.com/article/10.1007/s10618-020-00710-y>`_

.. tip::
    
    InceptionTime is applied to both univariate and multivariate times series classification.

------------

N-Beats
===============
N-Beats is a deep neural architecture based on backward and forward residual links and a very deep stack of fully-connected layers. The architecture has a number of desirable properties, being interpretable, applicable without modification to a wide array of target domains, and fast to train.

For more information, please refer to the paper `N-BEATS: Neural basis expansion analysis for interpretable time series forecasting <https://arxiv.org/abs/1905.10437>`_

.. tip::

    N-Beats is applied to both univariate and multivariate times series forecasting.

------------

VAE
=====
Variational AutoEncoder (VAE), is an unsupervised deep learning generative model, which can model the distribution of the training data. It comes from the Bayesian inference, and consists of an encoder, latent distribution, and a decoder. The principle is a simple distribution (such as a Gaussian distribution) with known parameters and superimposable characteristics can theoretically fit any distribution by combining with neural networks.

.. tip::
    
    VAE is applied to both univariate and multivariate times series anomaly detection.

--------

Neural Architecture Search
*****************************
Since AlexNet won the 2012 ImageNet competition, deep learning has made breakthroughs in many challenging tasks and fields. In addition to AlexNet,
e.g., VGG, Inception, ResNet, Transformer, GPT and so on have been proposed and widely used in industry and academia. And then, behind all these great networks,
it is the crystallization of the experience of countless human experts. Consequently, neural architecture search (NAS) has emerged as a promising tool to alleviate human efforts in this trial-and-error design process.

HyperTS relies on the basic capabilities provided by Hypernets (``Hpyer Model`` + ``Search Strategy`` + ``Estimation Strategy``), to build ``Search Space`` for time series tasks based on NAS powerful expression capabilities.

.. tip::
    NAS is applied to uni/multi-variate forecasting, classification, and regression.