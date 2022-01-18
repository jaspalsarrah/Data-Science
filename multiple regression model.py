# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 09:35:23 2021

@author: jsarrah
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn

my_data = pd.read_csv('Gapminder.csv')

# convert to numeric format
my_data["lifeexpectancy"] = pd.to_numeric(my_data["lifeexpectancy"],errors="coerce")
my_data["incomeperperson"] = pd.to_numeric(my_data["incomeperperson"],errors="coerce")
my_data["co2emissions"] = pd.to_numeric(my_data["co2emissions"],errors="coerce")
my_data["urbanrate"] = pd.to_numeric(my_data["urbanrate"],errors="coerce")
my_data["employrate"] = pd.to_numeric(my_data["employrate"],errors="coerce")
my_data["polityscore"] = pd.to_numeric(my_data["polityscore"],errors="coerce")


# listwise deletion of missing values
sub1 = my_data[['lifeexpectancy', 'incomeperperson', 'co2emissions', "urbanrate", "employrate", "polityscore" ]].dropna()

# first order (linear) scatterplot
scat1 = seaborn.regplot(x="incomeperperson", y="lifeexpectancy", scatter=True, data=sub1)
plt.xlabel("Income per person")
plt.ylabel("Life expectancy")

# fit second order polynomial
# run the 2 scatterplots together to get both linear and second order fit lines
scat2 = seaborn.regplot(x="incomeperperson", y="lifeexpectancy", scatter=True, order=2, data=sub1)
plt.xlabel('Income per person')
plt.ylabel('Life expectancy')

# centre quantitative IVs for regression analysis
sub1['incomeperperson_c'] = (sub1['incomeperperson'] - sub1['incomeperperson'].mean())
sub1['co2emissions_c'] = (sub1['co2emissions'] - sub1['co2emissions'].mean())
sub1['urbanrate_c'] = (sub1['urbanrate'] - sub1['urbanrate'].mean())
sub1['employrate_c'] = (sub1['employrate'] - sub1['employrate'].mean())
sub1['polityscore_c'] = (sub1['polityscore'] - sub1['polityscore'].mean())
sub1[["incomeperperson_c", "co2emissions_c", "urbanrate_c", "employrate_c", "polityscore_c"]].describe()

# linear regression analysis
print ("OLS linear regression model for the association between income per person and life expectancy")
reg1 = smf.ols('lifeexpectancy ~ incomeperperson_c', data=sub1).fit()
print (reg1.summary())

# Polynomial(quadratic) regression analysis
print ("OLS quadratic regression model for the association between income per person and life expectancy")
reg2 = smf.ols('lifeexpectancy ~ incomeperperson_c + I(incomeperperson_c**2)', data=sub1).fit()
print (reg2.summary())

# adding co2 emissions
print ("OLS regression model for the association between income per person/co2 emissions and life expectancy")
reg3 = smf.ols('lifeexpectancy ~ incomeperperson_c + I(incomeperperson_c**2) + co2emissions_c', data=sub1).fit()
print (reg3.summary())

# adding urban rate
print ("OLS regression model for the association between income per person/urban rate and life expectancy")
reg4 = smf.ols('lifeexpectancy ~ incomeperperson_c + I(incomeperperson_c**2) + urbanrate_c', data=sub1).fit()
print (reg4.summary())

# adding employ rate
print ("OLS regression model for the association between income per person/employ rate and life expectancy")
reg5 = smf.ols('lifeexpectancy ~ incomeperperson_c + I(incomeperperson_c**2) + employrate_c', data=sub1).fit()
print (reg5.summary())

# adding polity score
print ("OLS regression model for the association between income per person/polity score and life expectancy")
reg6 = smf.ols('lifeexpectancy ~ incomeperperson_c + I(incomeperperson_c**2) + polityscore_c', data=sub1).fit()
print (reg6.summary())

reg7 = smf.ols('lifeexpectancy ~ incomeperperson_c + I(incomeperperson_c**2) + co2emissions_c + urbanrate_c + employrate_c + polityscore_c', data=sub1).fit()
print (reg7.summary())

#Q-Q plot for normality
fig3=sm.qqplot(reg4.resid, line='r')

# simple plot of residuals
stdres=pd.DataFrame(reg4.resid_pearson)
plt.plot(stdres, 'o', ls='None')
l = plt.axhline(y=0, color='r')
plt.ylabel('Standardised Residual')
plt.xlabel('Observation Number')

# additional regression diagnostic plots
fig4 = plt.figure(figsize=(12,8))
fig4 = sm.graphics.plot_regress_exog(reg4, "urbanrate_c", fig=fig4)

# leverage plot
fig5=sm.graphics.influence_plot(reg4, size=8)
print(fig5)


