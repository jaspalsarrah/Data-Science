# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 09:56:09 2021

@author: jsarrah
"""
# Import libraries
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# Reading csv file using pandas dataframe
my_data = pd.read_csv("Gapminder.csv")

#setting variables you will be working with to numeric
my_data["polityscore"] = pd.to_numeric(my_data["polityscore"],errors="coerce")
my_data["lifeexpectancy"] = pd.to_numeric(my_data["lifeexpectancy"],errors="coerce")#
my_data["incomeperperson"] = pd.to_numeric(my_data["incomeperperson"],errors="coerce")#


# listwise deletion of missing values
sub1 = my_data[['lifeexpectancy', 'polityscore', 'incomeperperson']].dropna()
sub1.info()

# categorise incomeperperson variable using qcut function
# splits into 2 groups
sub1["incomeperperson"] = pd.qcut(sub1.incomeperperson, 2, labels=[0,1])
income_groups = sub1["incomeperperson"].value_counts(sort=False)
print(income_groups)

# categorise polityscore variable using qcut function
# splits into 2 groups
sub1["polityscore"] = pd.qcut(sub1.polityscore, 2, labels=[0,1])
polity_score_groups = sub1["polityscore"].value_counts(sort=False)
print(polity_score_groups)

# categorise life expectancy variable using qcut function
# splits into 2 groups at median
sub1["lifeexpectancy"] = pd.qcut(sub1.lifeexpectancy, 2, labels=[0,1])
life_expectancy_groups = sub1["lifeexpectancy"].value_counts(sort=False)
print(life_expectancy_groups)

# Setting variables to numeric as they have been changed to category after bins
sub1["incomeperperson"] = pd.to_numeric(sub1["incomeperperson"],errors="coerce")
sub1["polityscore"] = pd.to_numeric(sub1["polityscore"],errors="coerce")
sub1["lifeexpectancy"] = pd.to_numeric(sub1["lifeexpectancy"],errors="coerce")#

# logistic regression with income per person
lreg1 = smf.logit(formula = 'lifeexpectancy ~ incomeperperson', data = sub1).fit()
print (lreg1.summary())

# odd ratios with 95% confidence intervals
params = lreg1.params
conf = lreg1.conf_int()
conf['OR'] = params
conf.columns = ['Lower CI', 'Upper CI', 'OR']
print (np.exp(conf))

# logistic regression with polity score
lreg2 = smf.logit(formula = 'lifeexpectancy ~ polityscore', data = sub1).fit()
print (lreg2.summary())

# odd ratios with 95% confidence intervals
params = lreg2.params
conf = lreg2.conf_int()
conf['OR'] = params
conf.columns = ['Lower CI', 'Upper CI', 'OR']
print (np.exp(conf))

# logistic regression with income per person and polity score
lreg3 = smf.logit(formula = 'lifeexpectancy ~ incomeperperson + polityscore', data = sub1).fit()
print (lreg3.summary())

# odd ratios with 95% confidence intervals
params = lreg3.params
conf = lreg3.conf_int()
conf['OR'] = params
conf.columns = ['Lower CI', 'Upper CI', 'OR']
print (np.exp(conf))