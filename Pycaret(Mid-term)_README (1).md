
# PyCaret Library for MLOps

PyCaret is an open-source, low-code machine learning library in Python that automates machine learning workflows. It is an end-to-end machine learning and model management tool that speeds up the experiment cycle exponentially and makes you more productive.

PyCaret is essentially a Python wrapper around several machine learning libraries and frameworks such as scikit-learn, XGBoost, LightGBM, CatBoost, Optuna, Hyperopt, Ray, and many more.

 
Who should use PyCaret?

- Experienced Data Scientists who want to increase productivity.
- Citizen Data Scientists who prefer a low code machine learning solution.
- Data Science Professionals who want to build rapid prototypes.
- Data Science and Machine Learning students and enthusiasts.

Created on : April 2020 Version 1

Latest Version : 3.3.0


## Installation

A step-by-step guide to install PyCaret in Python.

#### Install via PyPi 

PyCaret is tested and supported on 64-bit systems with:

Python 3.8, 3.9, 3.10, and 3.11

Ubuntu 16.04 or later

Windows 7 or later

You can install PyCaret with Python's pip package manager:

Copy
pip install pycaret
PyCaret's default installation will not install all the optional dependencies automatically. Depending on the use case, you may be interested in one or more extras:

Copy
#### install analysis extras
pip install pycaret[analysis]

#### models extras
pip install pycaret[models]

#### install tuner extras
pip install pycaret[tuner]

#### install mlops extras
pip install pycaret[mlops]

#### install parallel extras
pip install pycaret[parallel]

#### install test extras
pip install pycaret[test]

#### install multiple extras together
pip install pycaret[analysis,models]

If you want to install everything including all the optional dependencies:

#### install full version
pip install pycaret[full]

#### Import pycaret

import pycaret
## Quickstart

### Pycaret.Regression Module

#### Setup
This function initializes the training environment and creates the transformation pipeline. Setup function must be called before executing any other function. It takes two required parameters: data and target. All the other parameters are optional.


#### load sample dataset
from pycaret.datasets import get_data
data = get_data('insurance')
#### Setup the data for Preprocessing
This function initializes the training environment and creates the transformation pipeline. Setup function must be called before executing any other function. It takes two required parameters: data and target. All the other parameters are optional.

from pycaret.regression import *
s = setup(data, target = 'charges', session_id = 145, log_experiment= True, normalize= True, experiment_name= 'insuranceRun')

#### Compare Models

This function trains and evaluates the performance of all the estimators available in the model library using cross-validation. The output of this function is a scoring grid with average cross-validated scores. Metrics evaluated during CV can be accessed using the get_metrics function. Custom metrics can be added or removed using add_metric and remove_metric function.

best = compare_models()
print(best)

#### Analyze Model Using Visulization

This function analyzes the performance of a trained model on the test set. It may require re-training the model in certain cases

evaluate_model(best)

plot_model(best, plot = 'residuals')

plot_model(best, plot = 'feature')

### Predictions

This function predicts prediction_label using the trained model. When data is None, it predicts label and score on the test set (created during the setup function).The evaluation metrics are calculated on the test set.

predict_model(best)
predictions.head()

### Save the model with complete pipeline

Finally, you can save the entire pipeline on disk for later use, using pycaret's save_model function.

save_model(best, 'my_best_pipeline')

### View all the Logged Experiments during ML Development

get_logs()
pull()

-------------------------------------------------------------------------------------------

### Pycaret.Clustering  Module

PyCaret’s Clustering Module is an unsupervised machine learning module that performs the task of grouping a set of objects in such a way that objects in the same group (also known as a cluster) are more similar to each other than to those in other groups. 

#### Setup
This function initializes the training environment and creates the transformation pipeline. Setup function must be called before executing any other function. It takes only one required parameter: data. All the other parameters are optional.

# load sample dataset
from pycaret.datasets import get_data
data = get_data('jewellery')

from pycaret.clustering import *
s = setup(data, normalize = True)

#### Create Model
This function trains and evaluates the performance of a given model. Metrics evaluated can be accessed using the get_metrics function. Custom metrics can be added or removed using the add_metric and remove_metric function. All the available models can be accessed using the models function.
#### Running Kmeans model with default number of clusters in Pycaret
kmeans = create_model('kmeans')
print(kmeans)

#### Running Kmeans model with number of cluster = 5
kmeans3 = create_model('kmeans', num_clusters= 5)
print(kmeans3)

#### Running Kmeans model with Affinity Propagation	Algorithm

kmeans2 = create_model('ap')
print(kmeans2)

#### Analyze Model

This function analyzes the performance of a trained model.

evaluate_model(kmeans)
evaluate_model(kmeans3)
evaluate_model(kmeans2)

#### Plot charts
plot_model(kmeans, plot = 'elbow')
plot_model(kmeans, plot = 'cluster')
plot_model(kmeans3, plot = 'cluster')

#### Assign model
This function assigns cluster labels to the training data, given a trained model.

#### Precidtion

kmeans_pred = predict_model(kmeans3, data=data_test)
kmeans_pred

#### Save the best Model for Deploymnet with all parameters
save_model(kmeans3, 'saved_kmeans_model')

#### Log all the Experiments with parameter,metrics as repository

logF = get_logs()
print(logF)
pull()
 



