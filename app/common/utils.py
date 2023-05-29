import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.model_selection import cross_val_score,KFold,StratifiedKFold

from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

import pickle
import asyncio

# 1. Problem Statement
# In this task we have to find the students' scores based on their study hours. The data has only two variables.
# Dependant variable : Scores
# Independant variable : Hours

# Data Gathering
scores_df=pd.read_excel(r'/home/agasti/Desktop/Assignment 4/app/data.xlsx')
print(scores_df.head())
print(scores_df.columns)
print(scores_df.info())

#Exploratory Data Analysis
print(scores_df.isna().sum())
print(scores_df['Hours'])
print(scores_df.describe())
print(scores_df.boxplot('Hours'))
plt.show()
print(scores_df['Hours'].skew())
print(scores_df.corr())
print(sns.heatmap(scores_df.corr(),annot=True))
plt.show()

# Model Training
x=scores_df.drop(['Scores'],axis=1)
y=scores_df['Scores']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

def model_building(algo,x,y):
    model = algo
    model.fit(x,y)
    return model

linear_reg = model_building(LinearRegression(),x_train,y_train)
print(linear_reg)

#Evaluation
# y_pred = linear_reg.predict(x_test)
# print(y_pred)

def evaluation(model,ind_var,y_act):
    pred=model.predict(ind_var)

    mse=mean_squared_error(y_act,pred)
    print('MSE : ',mse)

    mae=mean_absolute_error(y_act,pred)
    print('MAE : ',mae)

    r2_squared = r2_score(y_act,pred)
    print('R2_Score : ',r2_squared)

print('Test Data Evaluation'.center(50,'*'))
evaluation(linear_reg,x_test,y_test)

print('Train Data Evaluation'.center(50,'*'))
evaluation(linear_reg,x_train,y_train)

# loop = events.get_running_loop()
# asyncio.gather(asyncio.create_task(task1),asyncio.create_task(task2))

def save_model():
    with open(r'/home/agasti/Desktop/Assignment 4/app/model/model_pkl', 'wb') as file:
        lin_model = pickle.dump(linear_reg, file)
    return lin_model