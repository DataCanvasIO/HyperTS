Overview
########

HyperTS is an open source project created by the automatic data science platform provider `DataCanvas <https://www.datacanvas.com>`_ .



About HyperTS
===============
HyperTS is an automated machine learning (AutoML) and deep learning (AutoDL) tool which focuses on processing time series datasets. HyperTS belongs to the big family **DataCanvas AutoML Toolkits(DAT)** . It completely covers the full machine learning processing pipeline, consisting of data cleaning, preprocessing, feature engineering, model selection, hyperparameter optimization, result evalation and visalization. 



Why HyperTS
==================

HyperTS supports the following features: 

- **Multi-task support**

  HyerTS provides an uniform interface for various time series tasks, including forcasting, classification and regression.   

- **Multi-(co)variate support** 

  HyperTS supports both univariate and multivariate as input features for time series forecasting, as well as the covariates in the deep learning models.

- **Multi-mode support**
  
  STATS mode: For samll-scale datasets, HyperTS is able to quickly search optimal model and perform analysis by selecting the STATS mode, which contains several statistic models, like Prophet, ARIMA, and VAR.
  
  DL mode: For large-scale or complex datasets, users could select the DL mode, which provides several deep learning models (DeepAR, RNN, GRU, LSTM, LSTNet) to help build more robust neural network. Besides, the build-in GPU function significantly improves the time efficiency.

- **Powerful search strategies**
  
  HyperTS innovatively solves the hyperparameters optimization problem by collecting all hyperparameters over the full modeling process into one search space. The fundamental framework `Hypernets <https://github.com/DataCanvasIO/Hypernets>`_ of DAT provides multiple search algorithms (Adapting Grid Search, Monte Carlo Tree Search, Evolution Algorithm and Meta-learner) to ensure high efficiency search and optimization.
  
- **Abundant evaluation methods**

  To evaluate the trained model, HyperTS provides several performance matrics, including MSE, SMAPE, F1-score, accuracy and so on. Besides, forcasting curve with confidence intervals and controllable time scaling plot make the result more informative and better visualized. 
