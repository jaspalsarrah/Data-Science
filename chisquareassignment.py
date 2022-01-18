# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 09:43:09 2021

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

# categorise income per person variable based on customised splits using cut function
# splits into 5 groups (0-10000, 10000-20000, 20000-30000, 40000-50000)
sub2["incomeperperson"] = pd.cut(sub2.incomeperperson, [-1, 9999, 19999, 29999, 59999], labels=[1,2,3,4])
income_groups = sub2["incomeperperson"].value_counts(sort=False)
print(income_groups)

# categorise life expectancy variable based on customised splits using cut function
# splits into 2 groups at median
sub2["lifeexpectancy"] = pd.qcut(sub2.lifeexpectancy, 2, labels=[0,1])
life_expectancy_groups = sub2["lifeexpectancy"].value_counts(sort=False)
print(life_expectancy_groups)

# contingency table of observed counts
ct1=pd.crosstab(sub2['lifeexpectancy'], sub2['incomeperperson'])
print (ct1)

# column percentages
colsum=ct1.sum(axis=0)
colpct=ct1/colsum
print(colpct)

# chi-square
print ('chi-square value, p value, expected counts')
cs1= scipy.stats.chi2_contingency(ct1)
print (cs1)


#Post-hoc using Bonferroni. Seperate comparisons between different income groups.
# Comparison between income group 1 and 2
recode2 = {1: 1, 2: 2}
sub2['COMP1v2']= sub2['incomeperperson'].map(recode2)

ct2=pd.crosstab(sub2['lifeexpectancy'], sub2['COMP1v2'])
print (ct2)

colsum=ct2.sum(axis=0)
colpct=ct2/colsum
print(colpct)

print ('chi-square value, p value, expected counts')
cs2= scipy.stats.chi2_contingency(ct2)
print (cs2)

# Comparison between income group 1 and 3
recode3 = {1: 1, 3: 3}
sub2['COMP1v3']= sub2['incomeperperson'].map(recode3)

ct3=pd.crosstab(sub2['lifeexpectancy'], sub2['COMP1v3'])
print (ct3)

colsum=ct3.sum(axis=0)
colpct=ct3/colsum
print(colpct)

print ('chi-square value, p value, expected counts')
cs3= scipy.stats.chi2_contingency(ct3)
print (cs3)

# Comparison between income group 1 and 4
recode4 = {1: 1, 4: 4}
sub2['COMP1v4']= sub2['incomeperperson'].map(recode4)

ct4=pd.crosstab(sub2['lifeexpectancy'], sub2['COMP1v4'])
print (ct4)

colsum=ct4.sum(axis=0)
colpct=ct4/colsum
print(colpct)

print ('chi-square value, p value, expected counts')
cs4= scipy.stats.chi2_contingency(ct4)
print (cs4)

# Comparison between income group 2 and 3
recode5 = {2: 2, 3: 3}
sub2['COMP1v5']= sub2['incomeperperson'].map(recode5)

ct5=pd.crosstab(sub2['lifeexpectancy'], sub2['COMP1v5'])
print (ct5)

colsum=ct5.sum(axis=0)
colpct=ct5/colsum
print(colpct)

print ('chi-square value, p value, expected counts')
cs5= scipy.stats.chi2_contingency(ct5)
print (cs5)

# Comparison between income group 2 and 4
recode6 = {2: 2, 4: 4}
sub2['COMP1v6']= sub2['incomeperperson'].map(recode6)

ct6=pd.crosstab(sub2['lifeexpectancy'], sub2['COMP1v6'])
print (ct6)

colsum=ct6.sum(axis=0)
colpct=ct6/colsum
print(colpct)

print ('chi-square value, p value, expected counts')
cs6= scipy.stats.chi2_contingency(ct6)
print (cs6)

# Comparison between income group 3 and 4
recode7 = {3: 3, 4: 4}
sub2['COMP1v7']= sub2['incomeperperson'].map(recode7)

ct7=pd.crosstab(sub2['lifeexpectancy'], sub2['COMP1v7'])
print (ct7)

colsum=ct7.sum(axis=0)
colpct=ct7/colsum
print(colpct)

print ('chi-square value, p value, expected counts')
cs7= scipy.stats.chi2_contingency(ct6)
print (cs7)