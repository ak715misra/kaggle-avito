
# coding: utf-8

# In[16]:


# Load all the libraries necessary for the project 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
#import plotly.offline as py
#py.init_notebook_mode(connected=True)
#from plotly.offline import init_notebook_mode, iplot
#init_notebook_mode(connected=True)
#import plotly.graph_objs as go
#import plotly.offline as offline
#offline.init_notebook_mode()
#import plotly.tools as tls
color = sns.color_palette()
from sklearn import preprocessing, model_selection, metrics
#import lightgbm as lgb

# Avito data files are available in the "C:\akmisra\courses\MachineLearning\avito-data" directory.
import os
print(os.listdir("C:\\akmisra\\courses\\MachineLearning\\avito-data"))

# Results are saved as output.


# In[17]:


# Read the necessary data
print('Reading Data ...')
train = pd.read_csv("C:\\akmisra\\courses\\MachineLearning\\avito-data\\train.csv", parse_dates=["activation_date"])
print('training data size: ', train.shape)
test = pd.read_csv("C:\\akmisra\\courses\\MachineLearning\\avito-data\\test.csv", parse_dates=["activation_date"])
print('test data size: ', test.shape)
periods_train = pd.read_csv("C:\\akmisra\\courses\\MachineLearning\\avito-data\\periods_train.csv", parse_dates=["activation_date", "date_from", "date_to"])
print('periods_train data size: ', periods_train.shape)
periods_test = pd.read_csv("C:\\akmisra\\courses\\MachineLearning\\avito-data\\periods_test.csv", parse_dates=["activation_date", "date_from", "date_to"])
print('periods_test data size: ', periods_test.shape)
print('Finished Reading Data ...')


# In[18]:


# Check the data in train dataset
train.head()


# In[19]:


# test dataset should not have deal probability column
test.head()


# In[20]:


# Also check periods_train dataset
periods_train.head()


# In[21]:


# Training dataset overview
train.info()


# In[22]:


train.describe()


# In[26]:


# Use df.isnull.sum() to get the count of missing values in each column of df.
# Use df.isnull.count() to get the count of rows for each column in df 
total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/train.isnull().count()*100).sort_values(ascending=False)
missing_train_data = pd.concat([total, percent], axis=1, keys=["Total", "Percent"])
missing_train_data.head(10)


# In[27]:


# Also check if we have missing values in Periods_train dataset
total = periods_train.isnull().sum().sort_values(ascending=False)
percent = (periods_train.isnull().sum()/periods_train.isnull().count()*100).sort_values(ascending=False)
missing_train_data = pd.concat([total, percent], axis=1, keys=["Total", "Percent"])
missing_train_data.head()


# In[23]:


# data exploration - let us explore some of the data in the dataset
# deal probability is our target variable with float value between 0 and 1
# histogram and distribution of deal_probability
plt.figure(figsize = (12, 8))
sns.distplot(train['deal_probability'], bins=100, kde=False)
plt.xlabel('likelihood that ad actually sold something', fontsize=12)
plt.title('Histogram of likelihood that ad sold something')
plt.show()

plt.figure(figsize = (12, 8))
plt.scatter(range(train.shape[0]), np.sort(train.deal_probability.values))
plt.ylabel('likelihood that ad actually sold something', fontsize=12)
plt.title('Distribution of likelihood that ad sold something')
plt.show()


# The plots show that almost 1000000 ads had a probability of 0 (means sold nothing), while few had a probability of 1, and the rest were in the middle.

