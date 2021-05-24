#Collecting Data

import pandas as pd 
# used for data analysis
import numpy as np 
# this library is used for scientific computation
import seaborn as sns 
# used for statistical plotting
import matplotlib.pyplot as plt
# to run this library in jupiter notebook
import math

dataset_SUV = pd.read_csv("suv_data.csv")
print(dataset_SUV, end=" ")

print("No. of userID in original data:" + str(len(dataset_SUV.index)),  end=" ")

#How many people actually purachsed
sns.countplot(x="Purchased", data=dataset_SUV)

#How many male and female purchased SUV

sns.countplot(x="Purchased", hue="Gender", data=dataset_SUV)

#At what age most people tend to buy SUV

# size of countplot
sns.set(rc={'figure.figsize':(11.7,8.27)})

# countplot
sns.countplot(x="Age", data=dataset_SUV)

dataset_SUV.info()


# this will basically tell us what all values are null
null_values=dataset_SUV.isnull()
print(f"Null values : ",null_values)
dataset_SUV.isnull().sum()

# We'll analysis missing values with help of heat map

sns.heatmap(dataset_SUV.isnull(), yticklabels=False, cmap="viridis")


# Here we'll split the data into train subset and test subset.
# then we'll build a model on train data and predict the ouput on test data set

# Dependent variable (we have the discrete outcome)
X = dataset_SUV.iloc[:,[2,3]].values
# Independent variable (value data we need to predict)
y = dataset_SUV.iloc[:,4].values

# Now for splittibg data into testing and training subset we'll be using sklearn

from sklearn.model_selection import train_test_split

# type `train_test_split` and press shift+tab and you will able to see example of how train and test data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

from sklearn.preprocessing import StandardScaler

sc= StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# import Logistic Regression

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)

# Now we'll evaluate how our model has been performing
# we can calculate the accuraccy or we can calculate the classification report

from sklearn.metrics import classification_report

classification_report(y_test, predictions)

from sklearn.metrics import accuracy_score

x = accuracy_score(y_test, predictions)*100
print(f"Accracy : ",x)

from sklearn.metrics import confusion_matrix
y=confusion_matrix(y_test,predictions)
print(f"confusion matrix : ",y)