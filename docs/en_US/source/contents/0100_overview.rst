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

  HyerTS provides an uniform interface for various time series tasks, including forcasting, classification, regression, and anomaly detection.   

- **Multi-(co)variate support** 

  HyperTS supports both univariate and multivariate as input features for time series forecasting, as well as the covariates in the deep learning models.

- **Multi-mode support**
  
  STATS mode: For samll-scale datasets, HyperTS is able to quickly search optimal model and perform analysis by selecting the STATS mode, which contains several statistic models, like Prophet, ARIMA, and VAR.
  
  DL mode: For large-scale or complex datasets, users could select the DL mode, which provides several deep learning models (DeepAR, RNN, GRU, LSTM, LSTNet) to help build more robust neural network. Besides, the build-in GPU function significantly improves the time efficiency.

- **Powerful search strategies**
  
  HyperTS innovatively solves the hyperparameters optimization problem by collecting all hyperparameters over the full modeling process into one search space. The fundamental framework `Hypernets <https://github.com/DataCanvasIO/Hypernets>`_ of DAT provides multiple search algorithms (Adapting Grid Search, Monte Carlo Tree Search, Evolution Algorithm and Meta-learner) to ensure high efficiency search and optimization.
  
- **Abundant evaluation methods**

  After obtaining the trained model, HyperTS provides functions ``predict()`` and ``evaluate()`` to evaluate the model peformance. The output matrics include a variety of criterions like MSE, SMAPE, F1-score, accuracy, etc. Besides, function ``plot()`` will plot an interactive forcasting curves with confidence intervals, which makes the results more informative and better visualized. 


Feature Matrix
================

Below is the overview of all features and run modes of HyperTS:

.. csv-table:: 
   :stub-columns: 1
   :header: Category, Features, Current version, Future Version
   :widths: 5, 25, 5, 5
   
   Data cleaning, Repeated columns cleaning, √
   , Columns types correction, √
   , id column cleaning, √ 
   , Constant covariate columns cleaning, √
   , Deleting covariate columns with missing values, √
   , Deleting samples without targets, √
   Data preprocessing, TimeStamp impution, √
   , Missing value simple impution, √
   , Missing value average moving impution, √
   , Outliers processing, √
   , OrdinalEncoder, √
   , LabelEncoder, √
   , StandardScaler, √
   , MinMaxScaler, √
   , MaxAbsScaler, √
   , Log(x+1), √
   Dataset split, Split training dataset and test dataset in the order of time sequence, √
   Dataset creation , Create batches of inputs and targets by sliding window, √
   Model & Mode, Prophet —> STATS mode | univariate | forecasting, √
   , ARIMA —> STATS mode | univariate  | forecasting, √
   , VAR —> STATS mode | multivariate | forecasting, √
   , TSForest —> DL mode | univariate | classification, √
   , KNeighbors —> DL mode | uni/multi-variate | classification, √
   , Theta —> STATS mode | univariate | forecasting, , √
   , DeepAR —> DL mode | univariate  | forecasting | covariates , √
   , RNN —> DL mode | uni/multi-variate | forecasting/classification/regression | covariates, √
   , GRU —> DL mode | uni/multi-variate | forecasting/classification/regression  | covariates, √
   , LSTM —> DL mode | uni/multi-variate | forecasting/classification/regression | covariates, √
   , LSTNet —> DL mode | uni/multi-variate | forecasting/classification/regression  | covariates, √
   , InceptionTime —> DL mode | uni/multi-variate | classification  , √
   , N-Beats —> DL mode | uni/multi-variate | forecasting | covariates , √
   , VAE —> DL mode | uni/multi-variate | anomaly detection | covariates, √
   , NAS —> NAS | uni/multi-variate | forecasting/classification/regression  | covariates, √
   Evaluation methods, Train-Validation-Holdout, √
   , Rolling-Window-Evaluation, √
   Model ensemble, GreedyEnsemble, √
   Visualization, Forecasting curve, √
   , Forecasting trends and seasonality, , √
