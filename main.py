# -*- coding: utf-8 -*-

# -- Main --

import pandas as pd
train = pd.read_csv("csv/train.csv")
test = pd.read_csv("csv/test.csv")
gender = pd.read_csv("csv/gender_submission.csv")

train_shape = train.shape
train_isnull_check = train.isnull().sum()

import numpy as np
train.head()

# Selecting only numerical columns
numeric_train = train.select_dtypes(include=[np.number])

# Handling missing values (example: filling with zeros)
numeric_train = numeric_train.fillna(0)

# Calculating the correlation matrix
corr_matrix = numeric_train.corr()

import seaborn as sns
import matplotlib.pyplot as plt
# Creating a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.show()

# Histogram for Age
sns.histplot(train['Age'].dropna(), kde=True)
plt.title('Age Distribution')
plt.show()

# Distribution plots for Age by survival
sns.kdeplot(train[train['Survived'] == 1]['Age'], label='Survived')
sns.kdeplot(train[train['Survived'] == 0]['Age'], label='Did not survive')
plt.title('Age Distribution Among Survived and Non-Survived')
plt.legend()
plt.show()

# Bar charts for Ticket Class
sns.countplot(x='Pclass', hue='Survived', data=train)
plt.title('Passenger Count by Class and Survival')
plt.show()
