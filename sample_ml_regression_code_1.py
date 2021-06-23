# Import Standard Libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Embed/Render figures and graphs within the code editor rather than on separate pop-up windows/screens
%matplotlib inline

# Import ML Libaries 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn import svm
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Import Dataset from Scikit-Learn Library
from sklearn.datasets import load_iris

# Load Iris Dataset (X_features and y_targets)
iris = load_iris()

# X_features
print('X_features: Iris Dataset', '\n------------------------')
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
print(iris_df)

# y_targets
print('y_targets: Iris Dataset', '\n-----------------------')
print('y_targets values:', iris.target)
print('y_targets names:', iris.target_names)

def target_value_2_target_name(vals):
  temp = list()
  for i in vals:    
    if i == 0:
        temp.append('setosa')
    elif i == 1:
        temp.append('versicolor')
    elif i == 2:
        temp.append('virginica')

  return temp

# Convert 'y_targets' Numeric Values to Textual Labels
print('\ny_targets_names (updated): Iris Dataset', '\n---------------------------------------')
target_df = pd.DataFrame(target_value_2_target_name(iris.target), columns=['species'])
print(target_df)

# Fuse 'X_features' and 'y_targets' into a Single Dataframe
print('X_features && y_targets: Iris Dataset', '\n-------------------------------------')
iris_df = pd.concat([iris_df, target_df], axis= 1)  # Fuse along the vertical (or y) axis
print(iris_df)

# Basic Analytics of 'Iris' Dataset
print('Basic Analytics/Description: Iris Dataset', '\n-----------------------------------------')
iris_df.describe()
iris_df.info()
iris_df.shape

# Visualization of 'Iris' Dataset
plt.style.use('ggplot')
sns.pairplot(iris_df, hue='species')

# Variables 
X = iris_df.iloc[:, :-1]  # Returns all rows and all columns except the last column (species)
y = iris.target

# Split Iris Dataset into TRAIN and TEST portions
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size= 0.75, random_state= 32)
print('Shape of TRAIN X_features:', X_train.shape)
print('Shape of TEST X_features:', X_test.shape)

# Instantiating LinearRegression() Model
model = LinearRegression()
#model = svm.SVC()
#model = LogisticRegression()
model.fit(X_train, y_train)
model.predict(X_test)
pred = model.predict(X_test)

# Evaluating Model's Performance
print('Mean Absolute Error:', mean_absolute_error(y_test, pred))
print('Mean Squared Error:', mean_squared_error(y_test, pred))
print('Mean Root Squared Error:', np.sqrt(mean_squared_error(y_test, pred)))