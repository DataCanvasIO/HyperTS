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


Version 0.1.3
******************

Details of the HypertTS update are as follows:

- Tuning search space hyperparameters;

- Added report_best_trial_params;

- Fixed ARIMA to be compatible with version 0.12.1 and above;

- Fixed the pt issue of LSTNet;

- Simplified custom search space, task, timestamp, covariables and metircs can not be passed;

- Added OutliersTransformer, supported dynamic handling of outliers;

- Adjusted final train processing - lr, batch_size, epcochs and so on;
  
- Added time series meta-feature extractor;

- Added Time2Vec, RevIN, etc. layers;

- Added N-Beats time series forecasting model;

- Added InceptionTime time series classification model;

- Supported dynamic downsampling for time series forecasting;

- Refactored positive label inference method;

- Added neural architecture search mode;

- Fixed some known issues.


Version 0.1.4
******************

See Version 0.1.3.


Version 0.2.0
**************

Details of the HypertTS update are as follows:

- Supported time series **anomaly detection** task, and adapt to the full pipeline automation process;

- Added IForest anomaly detection model (stats mode);

- Added TSOneClassSVM anomaly detection model(stats mode);

- Added ConvVAE anomaly detection model(dl mode);

- Added realKnownCause anomaly detection dataset;

- Supported the visualization of anomaly detection results, and can analyze the anomaly location and severity;

- Compatible with Prophet version 1.1.1, now pip install hyperts for simultaneous successful prophet installation;

- Compatible with all versions of scipy;

- Added API documentation module;

- Supported for model persistence (saving and reloading trained models);

- In ```model.predict()``, fixed missing value handling;

- For the time series forecast task, the ```forecast``` function of DL model is calibrated;

- ```DLClassRegressSearchSpace``` was refactored for better adaptation to regression task;

- Extend ```InceptionTime``` to solve the regression task;

- Fixed some known issues;

- Thanks to **@Peter Cotton** for his contributions to hyperts.