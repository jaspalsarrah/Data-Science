# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 09:52:03 2021

@author: jsarrah
"""

import numpy as np
import pandas as pd

#Import statsmodel library to conduct statistical analysis
import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as multi

# Reading csv file using pandas dataframe
my_data = pd.read_csv("Gapminder.csv")

#setting variables you will be working with to numeric
my_data["incomeperperson"] = pd.to_numeric(my_data["incomeperperson"], errors="coerce")
my_data["lifeexpectancy"] = pd.to_numeric(my_data["lifeexpectancy"],errors="coerce")

#Drop observations with no income per person and no life expectancy
sub1 = my_data.dropna(subset=["incomeperperson", "lifeexpectancy"])
# Check for NaN in dataset
sub1.info()

# Copy of subset of data that will be used
sub2 = sub1.copy()

# categorise income per person variable based on customised splits using cut function
# splits into 5 groups (0-10000, 10000-20000, 20000-30000, 40000-50000, 50000-60000)
sub2["incomeperperson"] = pd.cut(sub2.incomeperperson, [-1, 9999, 19999, 29999, 39999, 49999, 59999])
income_groups = sub2["incomeperperson"].value_counts(sort=False, dropna=True)
print(income_groups)

#Data with null values dropped and income per person categorised for Anova
sub3 = sub2[['lifeexpectancy', 'incomeperperson']].dropna()

# using ols function for calculating the F-statistic and associated p value
model1 = smf.ols(formula='lifeexpectancy ~ C(incomeperperson)', data=sub3)
results1 = model1.fit()
print (results1.summary())

# Post hoc test as there were more than two explanatory variables
mc1 = multi.MultiComparison(sub3['lifeexpectancy'], sub3['incomeperperson'])
res1 = mc1.tukeyhsd()
print(res1.summary())