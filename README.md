# HSI_prediction

* data:  
  Yahoo finance
* prepare_data.py:  
  Data preparing and preprocesssing  
* feature_extra.py:  
  Features ranking according to their importance by Random Forest Classifier.  
* logistic_regression.py:  
  Bayesian Logistic Regression by tensorflow-probability to predict the next day's close price trend(upward or downward).
* model_* folders:  
  Storing the tensorflow graphs and variables with models built by different features (features with importance larger than 0.1,0.05, 0.04,0.03,etc.). 
