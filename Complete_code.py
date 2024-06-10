#!/usr/bin/env python
# coding: utf-8

# ### Importing necessary libraries

# In[1]:


import numpy as np
import pandas as pd
import re
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from matplotlib.colors import ListedColormap
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, hinge_loss, precision_score, recall_score )
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
import joblib


# ## Step 1: Data Collection

# In[2]:


# Loading the dataset
data = pd.read_csv('natural_disasters_dataset.csv')

# Displaying information about the Dataset
print(data.info())


# In[3]:


print(data.head(10))


# ### EDA before Preprocessing the Data

# #### i. Bar Plot for "Frequency of Disaster Types by Continent"

# In[4]:


# For this plot, we are grouping the data by 'Continent' and 'Disaster Type' and counting the frequency of the Disaster Type
disaster_counts = data.groupby(['Continent', 'Disaster Type']).size().unstack(fill_value=0)


plt.figure(figsize=(12, 6))
disaster_counts.plot(kind='bar', stacked=True, colormap='inferno')

# Setting x and y labels and Title for the plot
plt.xlabel('Continent')
plt.ylabel('Frequency')
plt.title('Frequency of Disaster Types by Continent')

# Customizing the Legend
plt.legend(title='Disaster Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# Displaying the top 5 continents with the highest frequency of disasters
top_continents = disaster_counts.sum(axis=1).nlargest(5)
print("Continents where the Disasters occurred more, in descending order:")
print(top_continents)


# #### ii. Horizontal Bar Chart for "Distribution of Disaster Types"

# In[5]:


plt.figure(figsize=(14, 8))
disaster_type_counts = data['Disaster Type'].value_counts().sort_values(ascending=False)

sns.barplot(x=disaster_type_counts, y=disaster_type_counts.index)

# Displaying the frequency count
for index, value in enumerate(disaster_type_counts):
    plt.text(value, index, str(value), va='center', fontsize=10, color='black', ha='left')

# Setting x and y labels and Title for the plot
plt.title('Distribution of Disaster Types')
plt.xlabel('Count')
plt.ylabel('Disaster Type')
plt.show()


# #### iii. Correlation Analysis - HeatMap

# In[6]:


# Visualizing correlation HeatMap for Numerical features
plt.figure(figsize=(16, 10))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='magma', fmt='.2f')

# Setting Title for the plot
plt.title('Correlation Heatmap')
plt.show()


# #### iv. Time series analysis of top 5 disasters

# In[7]:


import seaborn as sns

# Time series analysis for the count of each top 5 disaster types over the years
plt.figure(figsize=(14, 8))

# Filter the data for the top 5 disaster types
top_disaster_types = data['Disaster Type'].value_counts().nlargest(5).index
filtered_data = data[data['Disaster Type'].isin(top_disaster_types)]

# Group the filtered data by 'Start Year' and 'Disaster Type' and calculate the count
disaster_type_counts = filtered_data.groupby(['Start Year', 'Disaster Type']).size().unstack(fill_value=0)

# Plot time series for each top 5 disaster type
for disaster_type in disaster_type_counts.columns:
    plt.plot(disaster_type_counts.index, disaster_type_counts[disaster_type], label=disaster_type)

# Setting labels and title for the plot
plt.title('Time Series Analysis of Top 5 Disaster Types Over the Years')
plt.xlabel('Year')
plt.ylabel('Frequency')
plt.legend()
plt.show()


# ## Step 2: Data Preprocessing

# ### i. Checking for Missing values

# In[8]:


# Checking for Missing values
print("Null values:")
print(data.isnull().sum())


# ### ii. Handling Missing values

# In[9]:


data.replace('nan', np.nan, inplace=True)


# #### Imputing the Missing Numerical values with Mean and Categorical values with Mode

# In[10]:


numerical_cols = data.select_dtypes(include=np.number).columns
imputer = SimpleImputer(strategy='mean')
data[numerical_cols] = imputer.fit_transform(data[numerical_cols])

categorical_cols = data.select_dtypes(include='object').columns
imputer = SimpleImputer(strategy='most_frequent')
data[categorical_cols] = imputer.fit_transform(data[categorical_cols])


# In[11]:


print(data.isnull().sum())


# ### iii. Encoding Categorical variables 

# In[12]:


# Label encoding for categorical variables
label_encoder = LabelEncoder()
for col in categorical_cols:
    data[col] = label_encoder.fit_transform(data[col])


# ### EDA after Preprocessing the Data

# In[13]:


# Identify and remove single-valued columns
non_single_valued_columns = data.columns[data.nunique() > 1]
filtered_data = data[non_single_valued_columns]

# Visualizing correlation HeatMap for Numerical features
plt.figure(figsize=(16, 10))
correlation_matrix = filtered_data.corr()
sns.heatmap(correlation_matrix, annot= False, cmap='magma')

# Setting Title for the plot
plt.title('Correlation Heatmap')
plt.show()


# ## Step 3: Feature Selection

# #### Based on Mutal Information and Domain knowledge we selected the Features

# In[14]:


# Feature selection using mutual information
X = data.drop('Disaster Type', axis=1)  
y = data['Disaster Type']

# Based on Domain Knowledge we are including 'Year' feature
X['Year'] = X['Year'].astype(float)  
mutual_info = mutual_info_classif(X, y)
feature_importance = pd.Series(mutual_info, index=X.columns)

# Selecting top 10 features
selected_features = feature_importance.nlargest(10).index  

if 'Year' not in selected_features:
    selected_features = selected_features.append(pd.Index(['Year']))

# Printing the selected features
X_selected = X[selected_features]
print(X_selected.columns)


# In[15]:


# Loading the selected features into a new CSV file
selected_features = [
    'Year',  'Dis Mag Scale','Dis Mag Value', 'Country', 'Longitude', 'Latitude', 'Disaster Type'
]

data_selected = data[selected_features]


# In[16]:


# Save the new CSV file as preprocessed_data.csv
data_selected.to_csv('preprocessed_data.csv', index=False)


# In[17]:


# Load the preprocessed_data.csv file
data_selected = pd.read_csv('preprocessed_data.csv')


# In[18]:


data_selected.head(10)


# In[19]:


# Checking for missing values in the new csv file
print(data_selected.isnull().sum())


# ## Step 4: Model Development

# ### i. Random Forest

# In[20]:


# Separating feature set and target variable
X = data_selected.drop('Disaster Type', axis=1)
y = data_selected['Disaster Type']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=50,  # Number of trees in the forest
    max_depth=None,    # Maximum depth of the tree
    min_samples_split=2,  # Minimum number of samples required to split an internal node
    min_samples_leaf=1,   # Minimum number of samples required to be at a leaf node
    random_state=42
)

# Model Fitting
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

# Evaluation of the model
print("Random Forest Classifier:")
print("Accuracy:", accuracy_score(y_test, y_pred))


# ### ii. Support Vector Machine

# In[21]:


# Separating feature set and target variable
X = data_selected.drop('Disaster Type', axis=1)
y = data_selected['Disaster Type']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the Support Vector Machine model
svm_model = SVC(random_state=42)
svm_model.fit(X_train, y_train)

y_pred_svm = svm_model.predict(X_test)

# Evaluate the SVM model
print("Support Vector Machine (SVM):")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))


# ### iii. K- Nearest Neighbor

# In[22]:


# Separating feature set and target variable
X = data_selected.drop('Disaster Type', axis=1)
y = data_selected['Disaster Type']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the K-Nearest Neighbors model
knn_model = KNeighborsClassifier(
    n_neighbors=5,  # Number of neighbors to use
    weights='uniform',  # Weight function used in prediction
    algorithm='auto',  # Algorithm used to compute the nearest neighbors
)
knn_model.fit(X_train, y_train)

y_pred_knn = knn_model.predict(X_test)

# Evaluation of the KNN model
print("K-Nearest Neighbors (KNN):")
print("Accuracy:", accuracy_score(y_test, y_pred_knn))


# ### iv. Navie Bayes

# In[23]:


# Separate feature set and target variable
X = data_selected.drop('Disaster Type', axis=1)
y = data_selected['Disaster Type']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the Naive Bayes model
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

y_pred_nb = nb_model.predict(X_test)

# Evaluation of the Naive Bayes model
print("Naive Bayes:")
print("Accuracy:", accuracy_score(y_test, y_pred_nb))


# ## Step 5: Model Evaluation

# In[24]:


# Evaluating the performance of Random Forest
print("Random Forest Classifier Evaluation Metrics:")
print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Recall (Sensitivity):", recall_score(y_test, y_pred, average='weighted'))
print("Precision:", precision_score(y_test, y_pred, average='weighted'))
print("\n")

# Evaluating the performance of SVM
print("Support Vector Machine (SVM)-Evaluation Metrics:")
print("F1 Score:", f1_score(y_test, y_pred_svm, average='weighted'))
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Recall (Sensitivity):", recall_score(y_test, y_pred_svm, average='weighted'))
print("Precision:", precision_score(y_test, y_pred_svm, average='weighted'))
print("\n")

# Evaluating the performance of K-NN
print("K-Nearest Neighbor (K-NN)-Evaluation Metrics:")
print("F1 Score:", f1_score(y_test, y_pred_knn, average='weighted'))
print("Accuracy:", accuracy_score(y_test, y_pred_knn))
print("Recall (Sensitivity):", recall_score(y_test, y_pred_knn, average='weighted'))
print("Precision:", precision_score(y_test, y_pred_knn, average='weighted'))
print("\n")

# Evaluating the performance of Navie Bayes
print("Navie Bayes-Evaluation Metrics:")
print("F1 Score:", f1_score(y_test, y_pred_nb, average='weighted'))
print("Accuracy:", accuracy_score(y_test, y_pred_nb))
print("Recall (Sensitivity):", recall_score(y_test, y_pred_nb, average='weighted'))
print("Precision:", precision_score(y_test, y_pred_nb, average='weighted'))
print("\n")


# ## Step 6: Tuning

# #### As some of our models did not perform well, we checking whether our data is balaned or not

# In[25]:


# Checking the class distribution
print(data_selected['Disaster Type'].value_counts())


# #### Our data is not balanced, so to balance our data we are using Random Oversampler Technique

# In[26]:


# Separating feature set and target variable
X = data_selected.drop('Disaster Type', axis=1)
y = data_selected['Disaster Type']

# Initializing the RandomOverSampler
oversampler = RandomOverSampler(random_state=42)

# Fitting and applying the oversampling
X_resampled, y_resampled = oversampler.fit_resample(X, y)


# In[27]:


# Checking the new class distribution
print(pd.Series(y_resampled).value_counts())


# #### Now as we have a balanced data, we are processing from step 4 again

# In[28]:


# Splitting the resampled dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Standardizing/Scaling the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training the Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

y_pred_rf = rf_model.predict(X_test_scaled)

# Training the Support Vector Machine (SVM)
svm_model = SVC(kernel='linear', C=1.0, random_state=42)
svm_model.fit(X_train_scaled, y_train)

y_pred_svm = svm_model.predict(X_test_scaled)

# Training the K-Nearest Neighbors (KNN)
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)

y_pred_knn = knn_model.predict(X_test_scaled)

# Training the Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train_scaled, y_train)

y_pred_nb = nb_model.predict(X_test_scaled)


# In[29]:


# Evaluating the performance of Random Forest
print("Random Forest Classifier Evaluation Metrics:")
print("F1 Score:", f1_score(y_test, y_pred_rf, average='weighted'))
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Recall (Sensitivity):", recall_score(y_test, y_pred_rf, average='weighted'))
print("Precision:", precision_score(y_test, y_pred_rf, average='weighted'))
print("\n")

# Evaluating the performance of SVM
print("Support Vector Machine (SVM)-Evaluation Metrics:")
print("F1 Score:", f1_score(y_test, y_pred_svm, average='weighted'))
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Recall (Sensitivity):", recall_score(y_test, y_pred_svm, average='weighted'))
print("Precision:", precision_score(y_test, y_pred_svm, average='weighted'))
print("\n")

# Evaluating the performance of K-NN
print("K-Nearest Neighbor (K-NN)-Evaluation Metrics:")
print("F1 Score:", f1_score(y_test, y_pred_knn, average='weighted'))
print("Accuracy:", accuracy_score(y_test, y_pred_knn))
print("Recall (Sensitivity):", recall_score(y_test, y_pred_knn, average='weighted'))
print("Precision:", precision_score(y_test, y_pred_knn, average='weighted'))
print("\n")

# Evaluating the performance of Navie Bayes
print("Navie Bayes-Evaluation Metrics:")
print("F1 Score:", f1_score(y_test, y_pred_nb, average='weighted'))
print("Accuracy:", accuracy_score(y_test, y_pred_nb))
print("Recall (Sensitivity):", recall_score(y_test, y_pred_nb, average='weighted'))
print("Precision:", precision_score(y_test, y_pred_nb, average='weighted'))
print("\n")


# ## Performing Ensemble Techniques

# ### Hard voting Ensemble combining 4 models

# In[30]:


# Combining individual models to form a Ensemble Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
svm_model = SVC(kernel='linear', C=1.0, random_state=42)
knn_model = KNeighborsClassifier(n_neighbors=5)
nb_model = GaussianNB()

# Creating a hard voting classifier
ensemble_model = VotingClassifier(estimators=[
    ('rf', rf_model),
    ('svm', svm_model),
    ('knn', knn_model),
    ('nb', nb_model)
], voting='hard')

# Fitting the ensemble model
ensemble_model.fit(X_train_scaled, y_train)

y_pred_ensemble = ensemble_model.predict(X_test_scaled)

# Evaluate the ensemble model
print("\nEnsemble Model (Hard Voting):")
print("F1 Score:", f1_score(y_test, y_pred_ensemble, average='weighted'))
print("Accuracy:", accuracy_score(y_test, y_pred_ensemble))
print("Recall (Sensitivity):", recall_score(y_test, y_pred_ensemble, average='weighted'))
print("Precision:", precision_score(y_test, y_pred_ensemble, average='weighted'))


# ### Soft voting ensemble combining 4 models

# In[31]:


# Combining individual models with probability estimates to form an Ensemble Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
svm_model = SVC(kernel='linear', C=1.0, probability=True, random_state=42)
knn_model = KNeighborsClassifier(n_neighbors=5)
nb_model = GaussianNB()

# Creating a soft voting classifier
ensemble_model = VotingClassifier(estimators=[
    ('rf', rf_model),
    ('svm', svm_model),
    ('knn', knn_model),
    ('nb', nb_model)
], voting='soft')

# Fitting the ensemble model
ensemble_model.fit(X_train_scaled, y_train)

y_pred_ensemble = ensemble_model.predict(X_test_scaled)

# Evaluate the ensemble model
print("\nEnsemble Model (Soft Voting):")
print("F1 Score:", f1_score(y_test, y_pred_ensemble, average='weighted'))
print("Accuracy:", accuracy_score(y_test, y_pred_ensemble))
print("Recall (Sensitivity):", recall_score(y_test, y_pred_ensemble, average='weighted'))
print("Precision:", precision_score(y_test, y_pred_ensemble, average='weighted'))


# ### Visualization of Performance metrics using Grouped Bar chart

# In[32]:


import matplotlib.pyplot as plt
import numpy as np

# Models and their respective perfromance metrics
models = ['Random Forest', 'SVM', 'K-NN', 'Naive Bayes', 'Ensemble Hard-Voting', 'Ensemble Soft-Voting']
f1_scores = [0.9519, 0.6340, 0.9258, 0.6297, 0.91947, 0.93508 ]
accuracies = [0.9538, 0.6501, 0.9313, 0.6532, 0.92361, 0.93905 ]
recalls = [0.9538, 0.6501, 0.9313, 0.6532, 0.92361, 0.93905]
precisions = [0.9552, 0.6831, 0.9316, 0.7167, 0.92917, 0.94320]

# Grouped Bar chart
bar_width = 0.2
index = np.arange(len(models))

# Plotting the chart
fig, ax = plt.subplots(figsize=(14, 6))
bar1 = ax.bar(index - bar_width, f1_scores, bar_width, label='F1 Score')
bar2 = ax.bar(index, accuracies, bar_width, label='Accuracy')
bar3 = ax.bar(index + bar_width, recalls, bar_width, label='Recall')
bar4 = ax.bar(index + 2 * bar_width, precisions, bar_width, label='Precision')

# Settign up the Legend and Title
ax.set_xlabel('Models')
ax.set_ylabel('Scores')
ax.set_title('Model Evaluation Metrics')
ax.set_xticks(index + bar_width)
ax.set_xticklabels(models)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Show the plot
plt.show()


# #### * After visualizing the Grouped bar chart, we consider Random Forest to be our best Model <br> * We wanted to assess the generaliziability and performance of our model further. <br> * To achieve this we have done HyperParameter Tuning and Cross validation on our best model. <br>

# In[33]:


from sklearn.model_selection import GridSearchCV, StratifiedKFold

# Performing Hyperparameter Tuning using GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=StratifiedKFold(n_splits=5), scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

# Getting the best parameters
best_params = grid_search.best_params_
best_rf_model = grid_search.best_estimator_

# Print the best parameters
print("Best Hyperparameters for Random Forest:", best_params)


# In[34]:


# Loading Cross-Validation results into a Data frame
cv_results = pd.DataFrame(grid_search.cv_results_)
print("\nCross-Validation Results:")
print(cv_results[['param_n_estimators', 'param_max_depth', 'param_min_samples_split', 'param_min_samples_leaf', 'mean_test_score']])


# #### Perfroming evalution metrics again on the Best model after Hyperparameter Tuning

# In[35]:


# Evaluate the best model on the test set
y_pred_rf_tuned = best_rf_model.predict(X_test_scaled)

# Model Evaluation after Hyperparameter Tuning
print("Random Forest Classifier after Hyperparameter Tuning:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf_tuned))
print("F1 Score:", f1_score(y_test, y_pred_rf_tuned, average='weighted'))
print("Recall (Sensitivity):", recall_score(y_test, y_pred_rf_tuned, average='weighted'))
print("Precision:", precision_score(y_test, y_pred_rf_tuned, average='weighted'))


# In[36]:


import matplotlib.pyplot as plt
import numpy as np

# Metrics before hyperparameter tuning
metrics_before_tuning = {
    'Accuracy': 0.953822134150003,
    'F1 Score': 0.9519389979386549,
    'Recall': 0.953822134150003,
    'Precision': 0.9551815103634058
}

# Metrics after hyperparameter tuning
metrics_after_tuning = {
    'Accuracy': 0.953822134150003,
    'F1 Score': 0.9519389979386549,
    'Recall': 0.953822134150003,
    'Precision': 0.9551815103634058
}

metrics_names = list(metrics_before_tuning.keys())
values_before_tuning = list(metrics_before_tuning.values())
values_after_tuning = list(metrics_after_tuning.values())

bar_width = 0.35
index = np.arange(len(metrics_names))

fig, ax = plt.subplots()
bar1 = ax.bar(index, values_before_tuning, bar_width, label='Before Tuning')
bar2 = ax.bar(index + bar_width, values_after_tuning, bar_width, label='After Tuning')

ax.set_xlabel('Metrics')
ax.set_ylabel('Values')
ax.set_title('Random Forest Classifier Evaluation Metrics Before and After Hyperparameter Tuning')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(metrics_names)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.show()


# #### After visualzing the above graph we can observe that there is no change in the performance metrics of our best model, before and after hyper parameter tuning
# 
# #### Hence, we conclude that Our Best Model, Random forest's best performance metrics are <br>
# <b>Accuracy =  0.953822134150003<br>
#     F1 Score = 0.9519389979386549<br>
#     Recall =  0.953822134150003<br>
#     Precision = 0.9551815103634058</b>

# ## Step 7: Saving the Model

# In[37]:


# Fitting the Random Forest Classifier
rf_model.fit(X_train_scaled, y_train)

# Saving the trained Random Forest model to a file
joblib.dump(rf_model, 'random_forest_model.joblib')

# Evaluating the model
y_pred_rf = rf_model.predict(X_test_scaled)
print("Random Forest Classifier:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))


# In[38]:


# Initializing the RandomOverSampler
oversampler = RandomOverSampler(random_state=42)

# Fitting and applying the oversampling
X_resampled, y_resampled = oversampler.fit_resample(X, y)

# Saving the oversampler to a file
joblib.dump(oversampler, 'oversampler.joblib')


# In[ ]:




