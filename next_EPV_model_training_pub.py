# -*- coding: utf-8 -*-
"""
Training XGBoost model to predict the EPV added of the next event in a sequence of two successful passes
This code is written in blocks delimited by #%%

@author: Rob Oakley
"""

import numpy as np
import pandas as pd

#if you have SKillcorner API access, you can replace this with your credentials
username = 'username'
password = 'password'

from skillcorner.client import SkillcornerClient
client = SkillcornerClient(username=username, password=password)

import Skillcorner_IO_pub as skio

#load in your desired EPV grid
epv = np.loadtxt('EPV_grid.csv', delimiter = ',')


#load in XGBoost regressor
from xgboost import XGBRegressor

#load in event data and get list of all unique match_id numbers
events = pd.read_parquet('D:\ASI_Research\Wyscout_Event\wyscout_events_updated.parquet')
matches_list = events['sk_match_id'].unique().tolist()

#for all matches in matches_list generate the event sequences for that match and concatenate them all into a large dataframe
sequences = pd.concat((skio.get_event_sequences(m,events, EPV = epv) for m in matches_list), ignore_index=True)

#restrict to only events preceded by two successful passes
data = sequences[pd.notna(sequences['EPV_1'])]


#%%
#This block splits our sequences into training and testing data
from sklearn.model_selection import train_test_split

X = data[['location_x', 'location_y', 'x_1', 'y_1', 'EPV_1', 'x_2', 'y_2', 'EPV_2',]]
y = data['EPV_added']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

#%%
#This block trains a model with the default parameters
model = XGBRegressor()
model.fit(X_train, y_train)

y_predict = model.predict(X_test)

from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_absolute_error
print('RMSE: ', root_mean_squared_error(y_test, y_predict))
print('MAE: ', mean_absolute_error(y_test, y_predict))
#%%
#This block tries to find optimal parameters using randomized search
from sklearn.model_selection import RandomizedSearchCV
regressor  = XGBRegressor()
## Hyper Parameter Optimization
n_estimators = [100, 400, 800]
max_depth = [3, 6, 9, 12]
learning_rate=[0.05, 0.1, 0.2]
min_child_weight=[1, 10]

# Define the grid of hyperparameters to search
hyperparameter_grid = {
    'n_estimators': n_estimators,
    'max_depth':max_depth,
    'learning_rate':learning_rate,
    'min_child_weight':min_child_weight,
    }

# Set up the random search with 4-fold cross validation
random_cv = RandomizedSearchCV(regressor,
            param_distributions=hyperparameter_grid,
            cv=4, n_iter=10,
            scoring = 'neg_mean_absolute_error',n_jobs = 4,
            verbose = 5, 
            return_train_score = True,
            random_state=42)

random_cv.fit(X_train,y_train)

random_cv.best_estimator_
#%%
#Train a model using the optimal hyperparameters found in the prior block
model = XGBRegressor(learning_rate=0.05, max_depth=3, min_child_weight=1, n_estimators = 100)
model.fit(X_train,y_train)
y_predict = model.predict(X_test)

print('RMSE: ', root_mean_squared_error(y_test, y_predict))
print('MAE: ', mean_absolute_error(y_test, y_predict))
#%%
#add predicted epv add to dataframe
data_w_preds = data.copy()

data_w_preds['predicted_EPV_add'] = model.predict(data_w_preds[['location_x', 'location_y', 'x_1', 'y_1', 'EPV_1', 'x_2', 'y_2', 'EPV_2']])

#%%
#This block saves the data with predictions as a csv file for later use
data_w_preds.to_csv('data_w_pred_epv.csv', index = False)