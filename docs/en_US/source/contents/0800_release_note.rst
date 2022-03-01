Released Notes
===============

Version 0.1.0
**************

HyperTS is an AutoML and AutoDL tool which focuses on processing time series datasets. It supports the following features:

- Support the following data and tasks

  - forecasting, classification, and regression
  - univariate, multivariates, covariates

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

  - split by time sequence
  - split by moving window

- Model & mode: 

  - statistical models: Prophet、ARIMA、VAR、TSForest、KNeighbors
  - deep learning models: DeepAR、RNN、GRU、LSTM、LSTNet 

- Evaluation method: 

  - Train-Validation-Holdout
  
- Visualization:

  - human interactive plot
  - plot options: historical data, forecasting data and actural data
  - confidence intervals 
