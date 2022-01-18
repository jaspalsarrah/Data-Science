# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 14:13:05 2021

@author: jsarrah
"""

import numpy as np
import pandas as pd
import statsmodels.api
import statsmodels.formula.api as smf

# Reading csv file using pandas dataframe
my_data = pd.read_csv("Gapminderreg test.csv")

# Setting variables you will be working with to numeric
my_data["incomecentre"] = pd.to_numeric(my_data["incomecentre"],errors="coerce")
my_data["lifeexpectancy"] = pd.to_numeric(my_data["lifeexpectancy"],errors="coerce")

# Drop observations with no income per person and no life expectancy
sub1 = my_data.dropna(subset=["incomecentre", "lifeexpectancy"])
# Check for NaN in dataset
sub1.info()

# Copy of subset of data that will be used
sub2 = sub1.copy()

# Mean of income per person after centre (mean=0)
income_mean = sub2.mean()
print(income_mean)

print ("OLS regression model for the association between income per person and life expectancy")
reg1 = smf.ols('lifeexpectancy ~ incomeperperson', data=sub2).fit()
print (reg1.summary())