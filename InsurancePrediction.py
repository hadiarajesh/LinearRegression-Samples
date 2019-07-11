import pandas as pd
import numpy as np
import seaborn as seaborn
import matplotlib.pyplot as plot
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

dataset = pd.read_csv('D://Rajesh/ML/Dataset/insurance.csv')

# Plotting to analyse region-wise data
plot.figure(figsize=(15,10))
plot.tight_layout()
region_count = dataset['region'].value_counts()
seaborn.barplot(region_count.index, region_count.values, alpha=0.9)
plot.show()

# Using One-Hot Encoding to handle categorical data
dummy_cat_df = pd.get_dummies(dataset, columns=['sex', 'smoker', 'region'], drop_first=True)
#print(dummy_cat_df.columns.values)

# X contains independent variable and y contains dependent variable.
X = dummy_cat_df.drop('expenses', axis=1).values
y = dummy_cat_df['expenses'].values

# Splitting data to 80% as training and 20% as test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Initializing LinearRegression object and training it on training data
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Printing Intercept
print('Intercept is : ', regressor.intercept_)

# Getting column names to use while printing co-efficient
column_names = dummy_cat_df.drop('expenses', axis=1).columns.values

# Printing Coefficient
coef_df = pd.DataFrame(regressor.coef_, column_names, columns=['Coefficient'])
print(coef_df)

# Predicting output based on test data
y_predict = regressor.predict(X_test)

# Printing top 25 Actual vs Predicted value
df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_predict.flatten()})
print(df.head(25))

# Getting top 25 values of Actual vs Predicted data and plotting it
df1 = df.head(25)
df1.plot(kind='bar', figsize=(15, 10))
plot.title('Actual vs Predicted Value of Expense')
plot.grid(which='major', linestyle='-', linewidth='0.2', color='green')
plot.show()

print('Mean absolute error : ', metrics.mean_absolute_error(y_test, y_predict))
print('Mean squared error : ', metrics.mean_squared_error(y_test, y_predict))
print('Root mean squared error : ', np.sqrt(metrics.mean_squared_error(y_test, y_predict)))
