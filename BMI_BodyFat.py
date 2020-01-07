#!/usr/bin/env python
# coding: utf-8

# # Analysis of Body Mass Index and Body Fat Percentage
# 

# # Table of contents
# This report is organized in the following order:-
# 
# 1. [Overview](#Overview)
# This section provides a general intoduction and describes the dataset and theory used behind it while concurrently explaining the features in this dataset.
# 2. [Data Preparation](#Data-Preparation)
# This section covers the required data cleaning and data preparation steps.
# 3. [Data Exploration](#Data-Exploration)
# This section explores dataset features and how they are related.
# 4. [Statistical Modeling and Performance Evaluation](#Statistical-Modeling-and-Performance-Evaluation)
# This section first fits a complete multiple linear regression model. Then, we are performing a backward variable selection using the p-values to obtain a reduced model. After completing the above steps, we again perform another set of daignostic checks on the reduced model.
# 5. [Summary and Conclusions](#Summary-and-Conclusions)
# This section provides a summary of the work, conclude our observations and discuss about the results.
# 

# # Introduction

# The objective of this project is to  find out the relationships between ‘Body Fat Percentage’ and the 'Body Mass Index' that we will be calculating during the analysis.
# This is a comprehensive dataset that lists the estimates of the percentage of body fat determined by underwater weighing and various body circumference measurements for 252 men. This dataset was taken from http://staff.pubhealth.ku.dk/~tag/Teaching/share/data/Bodyfat.html.
# It was generally supplied by Dr. A. Garth Fisher who gave the authorization to freely distribute the data.
# 

# # Overview
# 
# Some experts thought BMI(body mass index) as the most accurate and simple way to determine the effect of weight on your health. In fact, most recent medical research uses BMI as an indicator of someone’s health status and disease risk. Some debate about which values on the BMI scale the thresholds for ‘underweight’, ‘overweight’ and ‘obese’ should be set. However, the followings are used for the criteria.
# BMI < 18.5 : underweight,
# 18.5 < BMI < 25 : optimal weight,
# 25 < BMI < 30 : overweight,
# BMI > 30 : obese.
# 
# Although, in this project we will just be determining if a person is Obese or not which is given by :-
# BMI < 25 : Physically Fit
# BMI > 25 : Overweight
# 
# The main goal of the project will be to determine a relation between body fat and BMI and to conclude which is a better measure.

# # Data Source
# 
# The dataset is being extracted from the above link providing just a single dataset named bodyfat.csv.
# the dataset contains 252 observations(a.k.a instances or records) and 15 decriptive(a.k.a independent) features(or attributes). 
# 

# # Objective
# 
# The goal for this project is to find the relation between 'Body Fat Percentage' and other predictors as well as 'Body Mass Index' and others.
# We will be creating a new feature(variable) named BMI.

# # Target Feature
# 
# Our target feature for this project will be 'BMI', which is calculated using the values of 'Weight' and 'Height' and is a continuous numerical feature, which further concludes that our project is on regression problem.

# # Descriptive Features
# 
# The variable descriptions below are from bodyfat.csv file:
# 
# Density : Density determined from underwater weighing
# Fat : Percent body fat from Siri’s (1956) equation
# Age : Age (years)
# Weight : Weight (lbs)
# Height : Height (inches)
# Neck : Neck circumference (cm)
# Chest: Chest circumference (cm)
# Abdomen : Abdomen circumference (cm)
# Hip : Hip circumference (cm)
# Thigh : Thigh circumference (cm)
# Knee : Knee circumference (cm)
# Ankle : Ankle circumference (cm)
# Biceps : Biceps (extended) circumference (cm)
# Forearm : Forearm circumference (cm)
# Wrist : Wrist circumference (cm)
# 
# Also, we will be calculating a new variable, named:
# BMI : Body Mass Index
# which will be created as a ratio of Weight(it should be in kgs hence we will be converting it to kgs from lbs while performing the calculations) to Height(it should be in m hence we will be converting it to m from inches while performing the calculations) squared for the measure of Body Mass Index in addition to the above variables.

# # Data Preparation

# # Preliminaries
# 
# 
# We would try to clean the dataset while taking into account the unusual values(such as negative vlue of age, weight etc), missing values, any categorical descriptive feature is encoded to be numeric, if the dataset has too many observations, we are supposed to work on a smaller random sample, though it is not the case for our dataset.

# In the first step, let us import all the common modules we might be using for this project.

# 

# In[7]:


# Importing modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import patsy
import io
import warnings
import requests
import re
###
warnings.filterwarnings('ignore')
###
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
plt.style.use("ggplot")


# We will be reading the bodyfat dataset with the help of the given .csv file. 

# In[8]:



c= pd.read_csv("Bodyfat.csv")


# We will be specifying the attribute names and will read in the data before displaying the random 10 rows from the dataset.

# In[9]:


# SPECIFYING THE ATTRIBUTES NAME
attributeNames = [
'Density determined from underwater weighing',
'Percent body fat from Siris (1956) equation',
'Age (years)'
'Weight (lbs)',
'Height (inches)',
'Neck circumference (cm)',
'Chest circumference (cm)',
'Abdomen 2 circumference (cm)',
'Hip circumference (cm)',
'Thigh circumference (cm)',
'Knee circumference (cm)',
'Ankle circumference (cm)',
'Biceps (extended) circumference (cm)',
'Forearm circumference (cm)',
'Wrist circumference (cm)',
]

# READ IN DATA 
url1 = pd.read_csv("Bodyfat.csv", sep = ',', names = attributeNames,header = None)

# DISPLAYING RANDOMLY SELECTED 10 ROWS
c.sample(10, random_state=999)






# # DATA CLEANING AND THE TRANSFORMATION
# 

# We need to confirm that the feature types match the decription provided in the given documentation

# In[10]:


print(f"Shape of the dataset is {c.shape} \n")
print(f"Data types are below where 'object' indicates a string type: ")
print(c.dtypes)


# # CHECKING FOR MISSING VALUES

# In[11]:


print(f"Missing values are:")
print(c.isnull().sum())


# In the given dataset, there are no attributes that contains any missing values.

# # SUMMARY STATISTICS 

# We are providing the summary of continous features below

# In[12]:


from IPython.display import display, HTML
display(HTML('<b>Table 1: Summary of continuous features</b>'))
c.describe(include='int64')


# In[13]:


display(HTML('<b>Table 2: Summary of categorical features</b>'))
print("There are no categorical objects")


# As there're no categorical features, we have no summary to display.

# In[14]:


c.head()


# We are supposed to add a new attribute named, BMI (Body Mass Index) which can be calculated by using the given formula:-
# 
# BMI = (Weight*0.4535)/((Height*0.0254)^2)
# where, 1 lb = 0.4535 kgs
#   and, 1 inch = 0.0254 m
# 

# In[15]:


print("Adding a new BMI column using the given formula ((Weight*0.4535)/((Height*0.0254)^2))")
print("where 1lb = 0.4535 kgs and 1 inch = 0.0254 m")
c['BMI'] = ((c['Weight']*(0.4535))/(c['Height']*(0.0254))**2)
# DISPLAYS THE FIRST 5 ROWS 
c.head()


# We do not have any indpendent variable which is of no use, as all the features we have are continous and numerical and are required to do the calculations and 'BMI' is dependent on each one of them.
# (We do not having any variables like ID which are of no use in the regression model)

# We do not have any columns that are required to be fixed neither do we have any categorical features to provie you with the unique values.

# We will be checking if any of our 15 attributes and the other attribute that we created contains any missing values and if they do, we can remove them and fix the dataset.
# Though this is quite a tiring process but we will still carry it out.

# In[16]:


print("We are checking for the missing values in the given dataset for the independent variables")
print("The given dataset has no mising values")
mask = (c['Density']== "" ) | (c['bodyfat'] == "") | (c['Age'] == "") | (c['Weight'] == "") | (c['Height'] == "") | (c['Neck'] == "") | (c['Chest'] == "") | (c['Abdomen'] == "") | (c['Hip'] == "") | (c['Thigh'] == "") | (c['Knee'] == "") | (c['Ankle'] == "") | (c['Biceps'] == "") | (c['Forearm'] == "") | (c['Wrist'] == "") |  (c['BMI'] == "")
mask.value_counts(normalize = True)*100
# print("The given dataset has no mising values")


# In[17]:


print("Our depenent variable in the given project is the BMI value ")


# We will be creating a new attribute named Result, which tells us if a given person has an BMI 0f less than 25 and greater than or equal to 25.
# If the perosn has he BMI less than 25, the person is considered fit and hence the 'Result' feature contains a 'true' value.
# Or if the value of BMI is greater than or equal to 25, the person is considered as overweight and hence the 'Result' feature contains a 'false' value.

# In[18]:


c['Result'] = c['BMI'] <=25
print("The Result here depicts if the person is Overweight or not")
print("Plotting the graphs for the two variables")


# # Data Exploration
# 

# Our dataset can now be considered as "clean" and is ready for visualiation and statistical modeling.
# 

# # Univariate Visualisation

# We will be plotting a bar chart for the attribute 'Result' which further determines if a person is overweight or not and is dependent on the value of BMI. This will help us in comparing between the two groups of people. 

# In[19]:


plotgraph = c['Result'].value_counts().plot(kind = 'bar')
plotgraph.set_xticklabels(plotgraph.get_xticklabels(), rotation = 90)
plt.tight_layout()
plt.title('Figure 1: Bar Chart of Result', fontsize = 15)
plt.show();


# We will be plotting a bar chart of the attribute 'Age' which further determine how does the value of BMI be varying with the 'Age' of the men.

# In[27]:


plotgraph = c['Age'].value_counts().plot(kind = 'bar')
plotgraph.set_xticklabels(plotgraph.get_xticklabels(), rotation = 30)
plt.tight_layout()
plt.title('Figure 2: Bar Chart for the Body Fat', fontsize = 15)
plt.show();

#  ax =c.plot.bar(x = 'Height', y = 'bodyfat', rot =0)


# In[30]:


plotgraph = c['Age'].value_counts().plot(kind = 'barh')
plotgraph.set_xticklabels(plotgraph.get_xticklabels(), rotation = 30)
plt.tight_layout()
plt.title('Figure 2: Bar Chart for the Body Fat', fontsize = 15)
plt.show();


# Let us plot a boxplot of bodyfat that tells us about its central value, and its variability.

# In[21]:


sns.boxplot(c['bodyfat']).set_title('Figure 1 : Box plot of Body Fat', fontsize = 15)
plt.show();


# And, we will be plotting a histogram of bodyfat which provides a visual interpretation of numerical data

# In[22]:


sns.distplot(c['bodyfat'], kde = True).set_title("Figure 2.1: Histogram for body fat", fontsize = 15)
plt.show()


# In[35]:


plt.hist(c['Age'],bins = [0,10,20,30,40,50,60,70,80,90,100], color = "orange", edgecolor = "blue", alpha = 1)
plt.title("Figure 2.2: Histogram for Age")
plt.xlabel("Age")
plt.ylabel("Number of People")
plt.grid(True)
plt.show()


# # Multivariate Visualisation

# # Scatterplot of Numeric Features and Age

# The scatterplot in Figure 4 shows a positive co-relation between the attribute 'bodyfat' and 'BMI'

# In[71]:


plt.scatter(c['bodyfat'], c['BMI'], alpha = 0.3)
plt.title("Figure 3: Scatterplot of bodyfat and BMI ", fontsize = 15)
plt.xlabel('bodyfat')
plt.ylabel('BMI')
plt.show();


# The boxplot between the attribute 'Result' and 'BMI' shows that people who are overweight have higher central value or median compared to those who ain't overweight though there is not much differnce.

# In[72]:


age_mask = (c['Result'])
age_index = c[age_mask]
sns.boxplot(age_mask, c['BMI']);
plt.title("Figure 4: Boxplot of Result and BMI", fontsize = 15)
plt.show();


# # Plots for more than 2 variables

# We are now plotting a scatterplot of Age by BMI, coloured by the Body Fat and all of them are related.
# Also, overall there is no pattern that we can observe but we can conclude that men below the age of 30 have the least body fat percentage.

# In[73]:


# c[(c.BMI <= 25)] = 'Overweight'
# Getting the index of those who work in the government
result_mask = (c['Result'])

# creating a dataframe of those who work in the government
result = c[result_mask]

# creating a scatterplot
sns.scatterplot(result['Age'], result['BMI'], hue = result['bodyfat'])
plt.title('Figure 8: Scatterplot of Age by BMI coloured by bodyfat', fontsize = 15);
plt.legend(loc = 'upper right')
plt.show();


# We will be plotting a scatterplot of Height by Weight, coloured by BMI and all of them are related. We can also see that it might hold a positive co-relation.

# In[74]:


# Getting the index of those who work in the government
gov_mask = (c['Result'])

# creating a dataframe of those who work in the government
gov = c[gov_mask]

# creating a scatterplot
sns.scatterplot(gov['Height'], gov['Weight'], hue = gov['BMI'])
plt.title('Figure 8: Scatterplot of Height by Weight coloured by BMI', fontsize = 15);
plt.legend(loc = 'upper right')
plt.show();


# # Statistical Modeling and Performance Evaluation

# # Full Model

# We will begin by fitting a multiple linear regression that predicts the "BMI" using all of the available features.
# Let's take a look at the clean data.

# In[75]:


c.head()


# 
# When constructing a regression model, we can manually add all the independent features but as there are quite a few, we can use some string function methods as you can see below:

# The formula string above works just fine with the Statsmodels module. The problem, however, is that we cannot do automatic variable selection with this formula. What we need for this purpose is "one-hot-encoding" of categorical features.

# In[76]:


formula_string_indep_vars = ' + '.join(c.drop(columns='BMI').columns)
formula_string = 'BMI ~ ' + formula_string_indep_vars
print('formula_string: ', formula_string)


# In the code chunk below, we first use the get_dummies() function in Pandas for one-hot-encoding of categorical features and then we construct a new formula string with the encoded features.

# In[77]:


c_encoded = pd.get_dummies(c, drop_first = True)
c_encoded.head()


# But, we don't have any categorical features and hence there is no difference in the output and it is same as earlier.

# In[78]:


formula_string_indep_vars_encoded = ' + '.join(c_encoded.drop(columns='BMI').columns)
formula_string_encoded = 'BMI ~ ' + formula_string_indep_vars_encoded
print('formula_string_encoded: ', formula_string_encoded)


# Again, no difference because of absence of categorical features.

# In[79]:


print("No categorical variables and hence the same result")


# Just to explore a bit more, we are adding two interaction terms to our model. Let's try adding the interaction of "Height" feature with "Neck" and "Chest".

# Now that we have defined our statistical model formula as a Python string, we fit an OLS (ordinary least squares) model to our encoded data.

# In[80]:


formula_string_encoded = formula_string_encoded + '+ Neck:Height + Chest:Height '
print('formula_string_encoded', formula_string_encoded)


# In[81]:


new_model = sm.formula.ols(formula=formula_string_encoded, data=c_encoded)
###
new_model_fitted = new_model.fit()
###
print(new_model_fitted.summary())


# Let's plot actual BMI values vs predicted values

# In[82]:


def plot_line(axis, slope, intercept, **kargs):
    xmin, xmax = axis.get_xlim()
    plt.plot([xmin, xmax], [xmin*slope+intercept, xmax*slope+intercept], **kargs)
    
# Creating scatter plot
plt.scatter(c_encoded['BMI'], new_model_fitted.fittedvalues, alpha=0.3);
plot_line(axis=plt.gca(), slope=1, intercept=0, c="red");
plt.xlabel('BMI');
plt.ylabel('Expected BMI');
plt.title('Figure : Scatterplot of age against expected BMI', fontsize=15);
plt.show();


# The full model has an adjusted R-squared value of 0.927, which means that about 92.7% of the variance, which is around 93% is explained by the model. From Figure , we observe that the model never produces a prediction above 40 even though the utmost value in the dataset is around 40 only, like what we can assume i=with that.Also, we can see a outlier at a BMI value of around 165 which is not practical. We will now check for the diagnostics for the model.

# # Full Model Diagnostic Checks

# In[83]:


sns.residplot(x=c_encoded['BMI'], y=new_model_fitted.fittedvalues);
plt.ylabel('residuals')
plt.title('Figure 10: Scatterplot of residuals', fontsize=15)
plt.show();


# From Figure , we see that the residuals appear mostly random and centered around 0. The exception is those above the BMI of 40 I think for whom the model predicts higher BMI. Let's also look at the histogram of the residuals.

# In[84]:


residuals = c_encoded['BMI'] - new_model_fitted.fittedvalues
plt.hist(residuals, bins = 20);
plt.xlabel('residual');
plt.ylabel('frequency');
plt.title('Figure 11: Histogram of residuals', fontsize=15);
plt.show();


# 
# 
# From Figure 11, the histogram of residuals looks somewhat symmetric but with some outlying values. This suggests that the residuals are somewhat normally distributed.

# In[85]:



# ## create the patsy model description from formula
model_reduced_fitted = smf.ols(formula = patsy_description, data = c_encoded).fit()
patsy_description = patsy.ModelDesc.from_formula(formula_string_encoded)

# initialize feature-selected fit to full model
linreg_fit = new_model_fitted

# do backwards elimination using p-values
p_val_cutoff = 0.05

## WARNING 1: The code below assumes that the Intercept term is present in the model.
## WARNING 2: It will work only with main effects and two-way interactions, if any.

print('\nPerforming backwards feature selection using p-values:')
pval_series = linreg_fit.pvalues.drop(labels='Intercept')

while True:

# #     # uncomment the line below if you would like to see the regression summary
# #     # in each step:


    pval_series = linreg_fit.pvalues.drop(labels='Intercept')
    pval_series = pval_series.sort_values(ascending=False)
    term = pval_series.index[0]
#     print(term)
    pval = pval_series[0]
#     print(pval)
    if (pval < p_val_cutoff):
        break
    term_components = term.split(':')
    print(f'\nRemoving term "{term}" with p-value {pval:.4}')
    if (len(term_components) == 1): ## this is a main effect term
        term_component = re.sub(r'\[.*\]', '', term_components[0])
        patsy_description.rhs_termlist.remove(patsy.Term([patsy.EvalFactor(term_component)]))    
    else: ## this is an interaction term
        patsy_description.rhs_termlist.remove(patsy.Term([patsy.EvalFactor(term_components[0]), 
                                                        patsy.EvalFactor(term_components[1])]))    
        
    linreg_fit = smf.ols(formula=patsy_description, data=c_encoded).fit()
    

    

print("\n***")
print(model_reduced_fitted.summary())
print("***")
print(f"Regression number of terms: {len(model_reduced_fitted.model.exog_names)}")
print(f"Regression F-distribution p-value: {model_reduced_fitted.f_pvalue:.4f}")
print(f"Regression R-squared: {model_reduced_fitted.rsquared:.4f}")
print(f"Regression Adjusted R-squared: {model_reduced_fitted.rsquared_adj:.4f}")


# In[86]:


# Creating scatter plot

plt.scatter(c_encoded['BMI'], model_reduced_fitted.fittedvalues, alpha=0.3);
plot_line(axis=plt.gca(), slope=1, intercept=0, c="red");
plt.xlabel('BMI');
plt.ylabel('Predicted BMI');
plt.title('Scatterplot of BMI against expected BMI for Reduced Model', fontsize=15);
plt.show(); 


# This model returns an Adjusted R-squared of 0.927, meaning the reduced model still explains about 92.7% of the variance, but with less variables. Looking at the p-values, they are all significant at the 5% level, as expected. We still have the problem as someone with BMI greater than 40 is being underestimated(Though it is highly unpracticle) We will now perform the diagnostic checks on this reduced model.

# In[87]:


sns.residplot(x=c_encoded['BMI'],  y=model_reduced_fitted.fittedvalues);
plt.ylabel('residuals')
plt.title('Scatterplot of residuals for Reduced Model', fontsize=15)
plt.show();


# The figure is very similar to earlier ones which depicts that the residuals appear mostly random and centered around 0

# In[88]:


residuals2 = c_encoded['BMI'] - model_reduced_fitted.fittedvalues
plt.hist(residuals2, bins = 20);
plt.xlabel('residual');
plt.ylabel('frequency');
plt.title('Histogram of residuals for Reduced Model', fontsize = 15)
plt.show();


# There is not much difference in the histogram either, though it looks a bit more symmetric and hence we conclude the same

# # Summary and Conclusions

# While only using the main effects,i.e, just using the main variables without any interaction, we still got a full model with an adjusted R-squared value of 92.7%, i.eabout 93%. Even after performing the backwards variables selection by removing the variables with a p-value cutoff value of 0.05, we were still able to maintain our results and the performance even with 8 less variables. With the residual scatter plots and residuals histograms, we got to onclude that there were not much significant violations of the assumptions of our regression model. This indicates that our statistical model is rather a valid one for practical purposes. Both Body Mass Index (BMI) and Body Fat Percent are measures of the body fat. For the BMI different response variables are used to identify the relationsip between each response variable and their own significant predictors to determine which BMI measure represents this dataset better. The final multiple linear regression model has an Adjusted R-squared value of 92.7% or about 93%, which is pretty high. As the ajusted R-squared is pretty high it appears that available features are a good predictor of BMI. Accurate measurement of body fat is inconvenient or costly and it is desirable to have easy methods of estimating body fat that are cost-effective and convenient. However, most of the body measures except for the Body Fat Percent and the Body Density are relatively easier to obtain for data collection. Body Mass Index is also calculated simply from weight and height of an individual. Conclusively, Body Mass Index is a more conveniet and less costly method for Body fat measures since it is strongly correlated with most body measures except Body Fat Percent and Density. One plausible problem with Body Mass Index regression is that there is a possiblity of multicollinearity among predictor variables since all body measures are highly correlated. However, What is the better or more accurate measure of human body fat remains to be answered with further medical researches.

# In[ ]:




