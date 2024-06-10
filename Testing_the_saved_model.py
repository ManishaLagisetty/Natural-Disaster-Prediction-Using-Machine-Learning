#!/usr/bin/env python
# coding: utf-8

# ### Loading the necessary libraries

# In[1]:


import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import joblib
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from matplotlib.colors import ListedColormap
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_classif
import numpy as np
import re
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score,recall_score,precision_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler


# ### Loading the new unseen dataset

# In[2]:


# Assuming 'new_data.csv' is your new, unseen data
new_data = pd.read_csv('new_unseen_dataset.csv')

# Checking for Missing values
print("Null values:")
print(new_data.isnull().sum())


# ### Preprocessing the new unseen dataset

# In[3]:


new_data.replace('nan', np.nan, inplace=True)

numerical_cols = new_data.select_dtypes(include=np.number).columns
imputer = SimpleImputer(strategy='mean')
new_data[numerical_cols] = imputer.fit_transform(new_data[numerical_cols])

categorical_cols = new_data.select_dtypes(include='object').columns
imputer = SimpleImputer(strategy='most_frequent')
new_data[categorical_cols] = imputer.fit_transform(new_data[categorical_cols])

# Label encoding for categorical variables
label_encoder = LabelEncoder()
for col in categorical_cols:
    new_data[col] = label_encoder.fit_transform(new_data[col])


# ### Feature Selection and Engineering - based on domain and mutal knowledge

# In[4]:


# Feature selection using mutual information
selected_features = ['Year', 'Dis Mag Scale', 'Dis Mag Value', 'Country', 'Longitude', 'Latitude', 'Disaster Type']
X_selected = new_data[selected_features]

# Saving the selected features into a new CSV file
X_selected.to_csv('new_unseen_preprocessed_data.csv', index=False)

# Loading the preprocessed_data.csv file
new_data_selected = pd.read_csv('new_unseen_preprocessed_data.csv')

# Displaying the first 10 rows of the loaded data
new_data_selected.head(10)


# ### Testing the Saved Model

# In[5]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib

# Load the oversampler and the trained Random Forest model
loaded_oversampler = joblib.load('oversampler.joblib')
loaded_model = joblib.load('random_forest_model.joblib')

# Assuming 'new_data.csv' is your new, unseen data
new_data = pd.read_csv('new_unseen_preprocessed_data.csv')

# Separate feature set and target variable
X_new = new_data.drop('Disaster Type', axis=1)
y_new = new_data['Disaster Type']

# Apply oversampler to the new data
X_new_resampled, y_new_resampled = loaded_oversampler.fit_resample(X_new, y_new)

# Splitting the resampled dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_new_resampled, y_new_resampled, test_size=0.2, random_state=42)

# Standardizing/Scaling the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Standardize/Scale the features using the previously defined scaler
X_new_scaled = scaler.transform(X_new_resampled)

# Make predictions using the loaded model
predictions = loaded_model.predict(X_new_scaled)

# Print classification report
print("Classification Report:")
print(classification_report(y_new_resampled, predictions))


# In[ ]:




