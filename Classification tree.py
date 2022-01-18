# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 12:04:09 2021

@author: jsarrah
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pylab as plt
import sklearn.metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report


# Reading csv file using pandas dataframe
my_data = pd.read_csv("Gapminder.csv")

# listwise deletion of missing values
sub1 = my_data.dropna()
sub1 = sub1.drop(["country"], axis=1)

# Convert all data to numeric
sub1[['incomeperperson','alcconsumption','armedforcesrate','breastcancerper100th','co2emissions',
'femaleemployrate','hivrate','internetuserate','lifeexpectancy','oilperperson','polityscore','relectricperperson','suicideper100th',
'employrate','urbanrate']] = sub1[['incomeperperson','alcconsumption','armedforcesrate','breastcancerper100th','co2emissions',
'femaleemployrate','hivrate','internetuserate','lifeexpectancy','oilperperson','polityscore','relectricperperson','suicideper100th',
'employrate','urbanrate']].apply(pd.to_numeric, errors="coerce")

# categorise life expectancy variable using qcut function
# splits into 2 groups
sub1["lifeexpectancy"] = pd.qcut(sub1.lifeexpectancy, 2, labels=[0,1])
life_expectancy_groups = sub1["lifeexpectancy"].value_counts(sort=False)


sub1["lifeexpectancy"] = pd.to_numeric(sub1["lifeexpectancy"],errors="coerce")#

sub1.dtypes
sub1 = my_data.dropna()
sub1.describe()

""" Modeling and Prediction """

#Split into training and testing sets

predictors = sub1[['incomeperperson','alcconsumption','armedforcesrate','breastcancerper100th','co2emissions',
'femaleemployrate','hivrate','internetuserate','lifeexpectancy','oilperperson','polityscore','relectricperperson','suicideper100th',
'employrate','urbanrate']]

targets = sub1.lifeexpectancy

pred_train, pred_test, tar_train, tar_test  =   train_test_split(predictors, targets, test_size=.4)

pred_train.shape
pred_test.shape
tar_train.shape
tar_test.shape

#Build model on training data
classifier=DecisionTreeClassifier()
classifier=classifier.fit(pred_train,tar_train)

predictions=classifier.predict(pred_test)

sklearn.metrics.confusion_matrix(tar_test,predictions)
sklearn.metrics.accuracy_score(tar_test, predictions)

#Displaying the decision tree
from sklearn import tree
#from StringIO import StringIO
from io import StringIO
#from StringIO import StringIO 
from IPython.display import Image
out = StringIO()
tree.export_graphviz(classifier, out_file=out)
import pydotplus
graph=pydotplus.graph_from_dot_data(out.getvalue())
Image(graph.create_png())

