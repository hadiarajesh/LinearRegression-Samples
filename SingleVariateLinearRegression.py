import pandas as pd
import numpy as np
import seaborn as seaBorn
import matplotlib.pyplot as plot
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Load csv file into dataset
dataset = pd.read_csv("D://PYTHON/DATASETS/weather.csv")
print(dataset.shape)
print(dataset.describe())

# Plotting MinTemp vs MaxTemp to analyse pattern
dataset.plot(x='MinTemp', y='MaxTemp', style='o')
plot.title('MinTemp vs MaxTemp')
plot.xlabel('MinTemp')
plot.ylabel('MaxTemp')
plot.show()

# Plotting MaxTemp to analyse average value
plot.figure(figsize=(15,10))
plot.tight_layout()
seaBorn.distplot(dataset['MaxTemp'])
plot.show()

# X contains Independent variable and y contains dependent variable
X = dataset['MinTemp'].values.reshape(-1,1)
y = dataset['MaxTemp'].values.reshape(-1,1)

# Splitting data to 80% as training and 20% as test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Initializing LinearRegression object and training it on training data
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Printing intercept and coefficient
print('Intercept is ', regressor.intercept_)
print('Coefficient', regressor.coef_)

# Predicting output based on test data
y_predict = regressor.predict(X_test)

# Printing Actual vs Predicted data
df = pd.DataFrame({'Actual' : y_test.flatten(), 'Predicted' : y_predict.flatten()})
print(df)

# Getting top 25 value and plotting it
df1 = df.head(25)
df1.plot(kind='bar', figsize=(15,10))
plot.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plot.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plot.show()

# Plotting predicted data with linear line
plot.scatter(X_test, y_test, color='grey')
plot.plot(X_test, y_predict, color='red', linewidth=2)
plot.show()

print('Mean absolute error : ', metrics.mean_absolute_error(y_test, y_predict))
print('Mean squared error : ', metrics.mean_squared_error(y_test, y_predict))
print('Root mean squared error : ', np.sqrt(metrics.mean_squared_error(y_test,y_predict)))
