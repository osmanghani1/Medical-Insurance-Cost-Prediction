#!/usr/bin/env python
# coding: utf-8

# # Importing the Dependencies

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import warnings 
warnings.filterwarnings('ignore')
from sklearn import metrics


# # Data collection and Analysis

# In[3]:


#loading the data from csv filr to pandas DF
insurance_dataset = pd.read_csv('insurance.csv')


# In[4]:


#first five rows of the data
insurance_dataset.head()


# In[5]:


#numbers of rows and columns
insurance_dataset.shape


# In[6]:


#getting more info about dataset
insurance_dataset.info()


# ## categorical features
# ### ->sex
# ### ->smoker
# ### ->region

# In[7]:


#checking for missing values
insurance_dataset.isnull().sum()


# # Data Analysis

# In[8]:


insurance_dataset.describe()


# In[9]:


#distribution of age value
sns.set()
plt.figure(figsize=(6,6))
sns.distplot(insurance_dataset['age'])
plt.title('Age Distribution')
plt.show()


# In[10]:


#Gender colum
plt.figure(figsize=(6,6))
sns.countplot(x='sex', data=insurance_dataset)
plt.title('Sex Distribution')
plt.show()


# In[11]:


insurance_dataset['sex'].value_counts()


# In[12]:


#bmi distribution
sns.set()
plt.figure(figsize=(6,6))
sns.distplot(insurance_dataset['bmi'])
plt.title('BMI Distribution')
plt.show()


# ## Normal BMI range --> 18.5 to 24.9

# In[13]:


#children colum
plt.figure(figsize=(6,6))
sns.countplot(x='children', data=insurance_dataset)
plt.title('children')
plt.show()


# In[14]:


insurance_dataset['children'].value_counts()


# In[15]:


#smoker colum
plt.figure(figsize=(6,6))
sns.countplot(x='smoker', data=insurance_dataset)
plt.title('Smoker')
plt.show()


# In[16]:


insurance_dataset['smoker'].value_counts()


# In[17]:


# region column
plt.figure(figsize=(6,6))
sns.countplot(x='region', data=insurance_dataset)
plt.title('region')
plt.show()


# In[18]:


insurance_dataset['region'].value_counts()


# In[19]:


# distribution of charges value
plt.figure(figsize=(6,6))
sns.distplot(insurance_dataset['charges'])
plt.title('Charges Distribution')
plt.show()


# # Data Pre-Processing
# ## Encoding the catagoreical features

# In[20]:


# encoding sex column
insurance_dataset.replace({'sex':{'male':0,'female':1}}, inplace=True)
# encoding 'smoker' column
insurance_dataset.replace({'smoker':{'yes':0,'no':1}}, inplace=True)
# encoding 'region' column
insurance_dataset.replace({'region':{'southeast':0,'southwest':1,'northeast':2,'northwest':3}}, inplace=True)


# # Splitting the Features and Target

# In[21]:


X = insurance_dataset.drop(columns='charges', axis=1)
Y = insurance_dataset['charges']


# In[22]:


print(X)


# In[23]:


print(Y)


# In[24]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)


# In[25]:


print(X.shape, X_train.shape, X_test.shape)


# # Model Training

# ## Linear Regression

# In[26]:


# loading the Linear Regression model
regressor = LinearRegression()


# In[27]:


regressor.fit(X_train, Y_train)


# # Model Evaluation
# 

# In[28]:


# prediction on training data
training_data_prediction =regressor.predict(X_train)


# In[29]:


# R squared value
r2_train = metrics.r2_score(Y_train, training_data_prediction)
print('R squared vale : ', r2_train)


# In[30]:


# prediction on test data
test_data_prediction =regressor.predict(X_test)


# In[31]:


# R squared value
r2_test = metrics.r2_score(Y_test, test_data_prediction)
print('R squared vale : ', r2_test)


# # Building a Predictive System

# In[33]:


input_data = (31,1,25.74,0,1,0)

# changing input_data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = regressor.predict(input_data_reshaped)
print(prediction)

print('The insurance cost is USD ', prediction[0])


# In[ ]:




