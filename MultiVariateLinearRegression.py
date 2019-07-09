import pandas as pd
import numpy as np
import seaborn as seaBorn
import matplotlib.pyplot as plot
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

dataset = pd.read_csv('D://PYTHON/DATASETS/wine_quality.csv')

# Check whether any column contains null value
print(dataset.isnull().any())

# If any column contains null value, fill it with previous value
dataset.fillna(method='ffill')

# X contains Independent variable and y contains dependent variable
X = dataset[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
             'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
             'pH', 'sulphates', 'alcohol']].values
y = dataset['quality'].values

# Plotting to analyse average value of quality
plot.figure(figsize=(15,10))
plot.tight_layout()
seaBorn.distplot(dataset['quality'])
plot.show()

# Splitting data to 80% as training and 20% as test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Initializing LinearRegression object and training it on training data
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Printing Intercept
print('Intercept is : ', regressor.intercept_)

# Getting column names excluding 'quality'
column_names = dataset.columns.values[:-1]

# Printing Coefficient
coef_df = pd.DataFrame(regressor.coef_, column_names, columns=['Coefficient'])
print(coef_df)

# Predicting output based on test data
y_predict = regressor.predict(X_test)

# Printing Actual vs Predicted value
df = pd.DataFrame({'Actual' : y_test.flatten(), 'Predicted' : y_predict.flatten()})
print(df)

# Getting top 25 values of Actual vs Predicted data and plotting it
df1 = df.head(25)
df1.plot(kind='bar', figsize=(15,10))
plot.title('Actual vs Predicted')
plot.grid(which='major', linestyle='-', linewidth='0.2', color='green')
plot.show()

print('Mean absolute error : ', metrics.mean_absolute_error(y_test, y_predict))
print('Mean squared error : ', metrics.mean_squared_error(y_test, y_predict))
print('Root mean squared error : ', np.sqrt(metrics.mean_squared_error(y_test, y_predict)))

