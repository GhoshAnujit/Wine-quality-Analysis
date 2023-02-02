#!/usr/bin/env python
# coding: utf-8

# In[44]:


#Importing the necessary libraries:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# In[45]:


#Importing the csv file
df = pd.read_csv("Wine_quality.csv")
#Printing the first 5 rows
df.head()


# In[46]:


#rows and columns
df.shape


# In[47]:


#checking for null values
df.isna().sum()


# In[48]:


#Particular rows with null values
print(df[df.isnull().any(axis = 1)])


# In[49]:


#renaming the "total sulphur dioxide" column for convinience
df.rename(columns = {'total sulfur dioxide':'TotalSulphurDioxide'}, inplace = True)


# In[55]:


#Removing the row with null value in the quality column
df = df.dropna(subset=['quality'])
#Replacing the null values with the mean values of the particular column
df.pH.fillna(df.pH.mean(), inplace=True)
df.TotalSulphurDioxide.fillna(df.TotalSulphurDioxide.mean(), inplace=True)


# In[56]:


df.isna().sum()
#All the null values are hence removed


# In[57]:


#Data analysis and visualization
#Statistical measures
df.describe()


# In[58]:


#number of values for each quality
sns.catplot(x ='quality', data = df , kind = 'count' )


# In[59]:


#volatile acidity and quality
plot = plt.figure(figsize = (6,6))
sns.barplot(x = 'quality', y = "volatile acidity", data = df)


# In[60]:


#citric acid and quality
plot = plt.figure(figsize = (6,6))
sns.barplot(x = 'quality', y = "citric acid", data = df)


# In[61]:


#residual sugar and quality
plot = plt.figure(figsize = (6,6))
sns.barplot(x = 'quality', y = "residual sugar", data = df)


# In[62]:


#chlorides and quality
plot = plt.figure(figsize = (6,6))
sns.barplot(x = 'quality', y = "chlorides", data = df)


# In[65]:


#free sulphur dioxide and quality
plot = plt.figure(figsize = (6,6))
sns.barplot(x = 'quality', y = "free sulfur dioxide", data = df)


# In[67]:


#TotalSulphurDioxide and quality
plot = plt.figure(figsize = (6,6))
sns.barplot(x = 'quality', y = "TotalSulphurDioxide", data = df)


# In[68]:


#density and quality
plot = plt.figure(figsize = (6,6))
sns.barplot(x = 'quality', y = "density", data = df)


# In[69]:


#pH and quality
plot = plt.figure(figsize = (6,6))
sns.barplot(x = 'quality', y = "pH", data = df)


# In[70]:


#sulphates and quality
plot = plt.figure(figsize = (6,6))
sns.barplot(x = 'quality', y = "sulphates", data = df)


# In[72]:


#alcohol and quality
plot = plt.figure(figsize = (6,6))
sns.barplot(x = 'quality', y = "alcohol", data = df)


# In[82]:


#Finding the correlation of all the columns to the quality column
correlation = df.corr()
correlation


# In[95]:


#constructing a heatmap to understand the correlation between the columns
plt.figure(figsize = (10,10))
sns.heatmap(correlation,cmap = "twilight_shifted", cbar = True, square = True, fmt = '.1f',annot = True, annot_kws = {"size":8})


# In[ ]:


#Data preprocessing


# In[91]:


#separate the data and the label
X = df.drop('quality',axis = 1)
X


# In[92]:


#Label Binarization
Y = df['quality'].apply(lambda y_value:1 if y_value>=7 else 0)
Y


# In[93]:


#Train and test Split
X_train, X_test, Y_train,Y_test = train_test_split(X,Y,test_size = 0.2, random_state = 3)


# In[94]:


print(Y.shape,Y_train.shape,Y_test.shape)


# In[96]:


#Model Training
#Random Forest Classifier model
model = RandomForestClassifier()


# In[97]:


model.fit(X_train, Y_train)


# In[98]:


#Model Evaluation
#accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[100]:


print("Accuracy: " , test_data_accuracy)


# In[105]:


#Building a predictive system
input_data = (7.9,0.6,0.06,1.6,0.069,15,59,0.9964,3.3,0.46,9.4)

#changing the input data to a numpy array
input_data_np = np.asarray(input_data)

#reshape the data as we are predicting the label for only one instance
input_data_reshaped = input_data_np.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if prediction[0] == 1:
    print("Good quality Wine")
else:
    print("Bad quality Wine")


# In[ ]:




