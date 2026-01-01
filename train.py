#!/usr/bin/env python
# coding: utf-8

# 🎓 Capstone Project 1 — California Housing Prediction
# 
# 1. Project Overview
# 1.1 Overview and Problem Statement
# 
# ## Import the required packages

# In[90]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


# ## Load Data

# In[91]:


df = pd.read_csv('housing.csv')
print(df.head())
print(df.shape)


# ## 2. Data Preparation and Exploratory Data Analysis (EDA)

# In[92]:


# 2. Data Preparation and Exploratory Data Analysis (EDA)

# Show the columns
print(df.columns)

# ocean_proximity is categorical variable
to_categorical = ['ocean_proximity']
df[to_categorical] = df[to_categorical].astype('category')

# housing_median_age and total_rooms are integer variables
to_integer = ['housing_median_age', 'total_rooms', 'households', 'population']
df[to_integer] = df[to_integer].astype('int64')

numerical_features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']

print(df.info())
print(df.describe())

# Locate NaN values in each column
nan_counts = df.isna().sum()
print("Number of NaN values in each column:")
print(nan_counts)




# In[93]:


# plot the histogram of the numerical features (in 2 columns)
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(20, 20))

for i, feature in enumerate(numerical_features):
    row = i // 2
    col = i % 2
    sns.histplot(df[feature], bins=30, kde=False, ax=axes[row, col])
    axes[row, col].set_title(f'Histogram of {feature}')
    axes[row, col].set_xlabel(feature)
    axes[row, col].set_ylabel('Frequency')

    plt.title(f'Histogram of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')

# plot the scatter plot of longitude and latitude
plt.figure(figsize=(10, 6))
sns.scatterplot(x='longitude', y='latitude', data=df, hue='median_house_value', palette='viridis')
plt.title('Scatter Plot of Longitude and Latitude with Median House Value')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

# plot heatmap of the correlation matrix of the numerical features
corr_matrix = df[numerical_features].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Numerical Features')
plt.show()

# Ocean Proximity Distribution
plt.figure(figsize=(10, 6))
ocean_counts = df['ocean_proximity'].value_counts()
plt.bar(ocean_counts.index, ocean_counts.values, color='steelblue')
plt.title('Distribution of Ocean Proximity Categories', 
          fontsize=14, fontweight='bold')
plt.xlabel('Ocean Proximity', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Box plots for median_house_value by ocean_proximity
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='ocean_proximity', y='median_house_value')
plt.title('House Value Distribution by Ocean Proximity', 
          fontsize=14, fontweight='bold')
plt.xlabel('Ocean Proximity', fontsize=12)
plt.ylabel('Median House Value ($)', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# # 3. Data Preprocessing

# In[94]:


# Split the data into training and testing sets
df_all_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df_all_train, test_size=0.25, random_state=42)

# Handle missing values in total_bedrooms using median from training set
median_bedrooms = df_train['total_bedrooms'].median()
df_train['total_bedrooms'] = df_train['total_bedrooms'].fillna(median_bedrooms)
df_all_train['total_bedrooms'] = df_all_train['total_bedrooms'].fillna(median_bedrooms)
df_val['total_bedrooms'] = df_val['total_bedrooms'].fillna(median_bedrooms)
df_test['total_bedrooms'] = df_test['total_bedrooms'].fillna(median_bedrooms)

y_train = df_train.median_house_value.values
y_all_train = df_all_train.median_house_value.values
y_val = df_val.median_house_value.values
y_test = df_test.median_house_value.values

df_train = df_train.drop('median_house_value', axis=1)
df_all_train = df_all_train.drop('median_house_value', axis=1)
df_val = df_val.drop('median_house_value', axis=1)
df_test = df_test.drop('median_house_value', axis=1)

dv = DictVectorizer(sparse=False)

train_dict = df_train.to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

all_train_dict = df_all_train.to_dict(orient='records')
X_all_train = dv.transform(all_train_dict)

val_dict = df_val.to_dict(orient='records')
X_val = dv.transform(val_dict)

test_dict = df_test.to_dict(orient='records')
X_test = dv.transform(test_dict)

print(dv.get_feature_names_out())


# ## 4. Model Training and Evaluation

# In[95]:


# 4. Model Training and Evaluation

# Train the model with Linear Regression (Baseline Model) 
baseline_model = LinearRegression()
baseline_model.fit(X_train, y_train)
baseline_predictions = baseline_model.predict(X_val)
baseline_rmse = np.sqrt(mean_squared_error(y_val, baseline_predictions))
print(f"Baseline Model has RMSE: {baseline_rmse}")

# Train the model with Lasso Regression with hyperparameters tuning
lasso_alpha = [0.001, 0.01, 0.1, 1, 10, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 40, 50, 60, 70, 80, 90, 100, 1000]
lasso_rmse = []
for alpha in lasso_alpha:
    lasso_model = Lasso(alpha=alpha)
    lasso_model.fit(X_train, y_train)
    lasso_predictions = lasso_model.predict(X_val)
    lasso_rmse.append(np.sqrt(mean_squared_error(y_val, lasso_predictions)))
    print(f"Lasso Model with alpha={alpha} has RMSE: {lasso_rmse[-1]}")

# Train the model with Ridge Regression with hyperparameters tuning
ridge_alpha = [0.001, 0.01, 0.1, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 30, 40, 50, 60, 70, 80, 90, 100, 1000]
ridge_rmse = []
for alpha in ridge_alpha:
    ridge_model = Ridge(alpha=alpha)
    ridge_model.fit(X_train, y_train)
    ridge_predictions = ridge_model.predict(X_val)
    ridge_rmse.append(np.sqrt(mean_squared_error(y_val, ridge_predictions)))
    print(f"Ridge Model with alpha={alpha} has RMSE: {ridge_rmse[-1]}")

# Best parameters
best_lasso_alpha = lasso_alpha[np.argmin(lasso_rmse)]
best_ridge_alpha = ridge_alpha[np.argmin(ridge_rmse)]
print(f"Best Lasso Alpha: {best_lasso_alpha}")
print(f"Best Ridge Alpha: {best_ridge_alpha}")

# Update the models with the best parameters
lasso_model = Lasso(alpha=best_lasso_alpha)
lasso_model.fit(X_train, y_train)
ridge_model = Ridge(alpha=best_ridge_alpha)
ridge_model.fit(X_train, y_train)


# In[ ]:


# Train the model with Neural Network (Keras)

# Convert sparse matrices to dense arrays for Keras (keep original sparse matrices for scikit-learn)
from scipy.sparse import issparse
if issparse(X_train):
    X_train_dense = X_train.toarray()
    X_val_dense = X_val.toarray()
    X_test_dense = X_test.toarray()
else:
    X_train_dense = X_train
    X_val_dense = X_val
    X_test_dense = X_test

# Use gradient clipping to prevent NaN
optimizer = Adam(clipnorm=1.0)

nn_model = Sequential()
nn_model.add(Dense(64, input_dim=X_train_dense.shape[1], activation='relu'))
nn_model.add(Dense(32, activation='relu'))
nn_model.add(Dense(1))
nn_model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=[keras.metrics.RootMeanSquaredError()])
nn_model.fit(X_train_dense, y_train, epochs=50, batch_size=32, validation_data=(X_val_dense, y_val))
test_results = nn_model.evaluate(X_test_dense, y_test, verbose=0)
nn_predictions = nn_model.predict(X_test_dense, verbose=0)
# Flatten predictions to 1D array
nn_predictions = nn_predictions.flatten()
# Check for NaN and calculate RMSE
if np.isnan(nn_predictions).any() or np.isnan(test_results).any():
    print("Warning: Model produced NaN. Using validation set for evaluation.")
    val_results = nn_model.evaluate(X_val_dense, y_val, verbose=0)
    nn_predictions_val = nn_model.predict(X_val_dense, verbose=0).flatten()
    nn_rmse = np.sqrt(mean_squared_error(y_val, nn_predictions_val))
    print(f"Neural Network Model Validation RMSE: {nn_rmse:.2f}")
else:
    nn_rmse = np.sqrt(mean_squared_error(y_test, nn_predictions))
    print(f"Neural Network Model Test RMSE: {nn_rmse:.2f}")


# ## Visualize the Results

# In[100]:


# Plot the predictions of each of the above models in separate subplots

# Create subplots for each model
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Baseline Model
axes[0, 0].scatter(y_val, baseline_predictions, label='Baseline Model')
axes[0, 0].plot(y_val, y_val, color='black', linestyle='--', label='x = y')
axes[0, 0].set_title('Baseline Model')
axes[0, 0].set_xlabel('Actual Values')
axes[0, 0].set_ylabel('Predicted Values')
axes[0, 0].legend()

# Lasso Model
axes[0, 1].scatter(y_val, lasso_predictions, label='Lasso Model')
axes[0, 1].plot(y_val, y_val, color='black', linestyle='--', label='x = y')
axes[0, 1].set_title('Lasso Model')
axes[0, 1].set_xlabel('Actual Values')
axes[0, 1].set_ylabel('Predicted Values')
axes[0, 1].legend()

# Ridge Model
axes[1, 0].scatter(y_val, ridge_predictions, label='Ridge Model')
axes[1, 0].plot(y_val, y_val, color='black', linestyle='--', label='x = y')
axes[1, 0].set_title('Ridge Model')
axes[1, 0].set_xlabel('Actual Values')
axes[1, 0].set_ylabel('Predicted Values')
axes[1, 0].legend()

# Neural Network Model
axes[1, 1].scatter(y_val, nn_predictions, label='Neural Network Model')
axes[1, 1].plot(y_val, y_val, color='black', linestyle='--', label='x = y')
axes[1, 1].set_title('Neural Network Model')
axes[1, 1].set_xlabel('Actual Values')
axes[1, 1].set_ylabel('Predicted Values')
axes[1, 1].legend()

plt.tight_layout()
plt.show()


# In[ ]:


# Test the models with the test set
baseline_predictions_test = baseline_model.predict(X_test)
lasso_predictions_test = lasso_model.predict(X_test)
ridge_predictions_test = ridge_model.predict(X_test)
nn_predictions_test = nn_model.predict(X_test_dense)

# Calculate the MSE of the test set
baseline_rmse_test = np.sqrt(mean_squared_error(y_test, baseline_predictions_test))
lasso_rmse_test = np.sqrt(mean_squared_error(y_test, lasso_predictions_test))
ridge_rmse_test = np.sqrt(mean_squared_error(y_test, ridge_predictions_test))
nn_rmse_test = np.sqrt(mean_squared_error(y_test, nn_predictions_test))

print(f"Baseline Model has RMSE: {baseline_rmse_test:.2f}")
print(f"Lasso Model has RMSE: {lasso_rmse_test:.2f}")
print(f"Ridge Model has RMSE: {ridge_rmse_test:.2f}")
print(f"Neural Network Model has RMSE: {nn_rmse_test:.2f}")


# In[103]:


# Understand the linear regression model parameters
coefficients = baseline_model.coef_
intercept = baseline_model.intercept_

print(f"Coefficients: {coefficients}")
print(f"Intercept: {intercept}")

# Rank the features by their importance
feature_importance = pd.DataFrame({
    'Feature': dv.get_feature_names_out(),
    'Importance': baseline_model.coef_
})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
print(feature_importance.round(2))

# Plot the feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance, orient='h')
plt.title('Feature Importance from Linear Regression Model')
plt.xlabel('Importance')


# ## Model Export and Scripting

# In[ ]:


# Use pickle to export the models: baseline_model, lasso_model, ridge_model, model
import pickle

with open('model.pkl', 'wb') as f:
    pickle.dump((dv, baseline_model, lasso_model, ridge_model, nn_model), f)

