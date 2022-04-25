Released Notes
===============

Version 0.1.0
**************

HyperTS is an AutoML and AutoDL tool which focuses on processing time series datasets. It supports the following features:

- Support the following data and tasks

  - Forecasting, classification, and regression
  - Univariate, multivariates, covariates

- Data cleaning:

  - Repeated columns cleaning  
  - Columns types correction  
  - `ID` column cleaning  
  - Constant covariate columns cleaning  
  - Deleting covariate columns with missing values  
  - Deleting samples without targets

- Data preprocessing: 

  - TimeStamp impution  
  - Missing value simple impution
  - Missing value average moving impution
  - Outliers processing
  - OrdinalEncoder
  - LabelEncoder
  - StandardScaler
  - MinMaxScaler
  - MaxAbsScaler
  - Log(x+1)

- Data split: 

  - Split training dataset and test dataset in the order of time sequence	
  
- Dataset creation	
  
  - Create batches of inputs and targets by sliding window
 

- Model & mode: 

  - Statistical models: Prophet、ARIMA、VAR、TSForest、KNeighbors
  - Deep learning models: DeepAR、RNN、GRU、LSTM、LSTNet 

- Evaluation method: 

  - Train-Validation-Holdout
  
- Visualization:

  - Human interactive plot
  - Plot options: historical data, forecasting data and actural data
  - Confidence intervals 


Version 0.1.2.1
******************

Details of the HypertTS update are as follows:

- Supports cross validation.

- Supports greedy ensemble.

- Supports time series forecasting data without timestamp column.

- Supports time series forecasting truncation training set to train.

- Supports time series forecasting of discrete data (no  fixed time frequency).

- Supports Fourier inference period.

- Supports non-invasive parameters tuning.

- Optimizes search space and architecture.

- Fixes some bugs.