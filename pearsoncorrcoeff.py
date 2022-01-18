# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 14:04:14 2021

@author: jsarrah
"""

#Import libraries
import pandas as pd
import numpy as np
import scipy.stats

# Reading csv file using pandas dataframe
my_data = pd.read_csv("Gapminder.csv")

#setting variables you will be working with to numeric
my_data["incomeperperson"] = pd.to_numeric(my_data["incomeperperson"],errors="coerce")
my_data["lifeexpectancy"] = pd.to_numeric(my_data["lifeexpectancy"],errors="coerce")

#Drop observations with no income per person and no life expectancy
sub1 = my_data.dropna(subset=["incomeperperson", "lifeexpectancy"])
# Check for NaN in dataset
sub1.info()

# Copy of subset of data that will be used
sub2 = sub1.copy()

print ('Association between income and lifeexpectancy')
print (scipy.stats.pearsonr(sub2['incomeperperson'], sub2['lifeexpectancy']))

