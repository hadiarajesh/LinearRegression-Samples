import pandas as pd
import numpy as np
import seaborn as seaborn
import matplotlib.pyplot as plot
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

dataset = pd.read_csv('D://PYTHON/DATASETS/insurance.csv')

# Plotting to analyse average age
plot.figure(figsize=(15,10))
plot.tight_layout()
seaborn.distplot(dataset['age'])
plot.show()

# Getting categorical data
cat_df = dataset.select_dtypes('object').copy()

# Plotting to analyse region-wise data
plot.figure(figsize=(15,10))
plot.tight_layout()
region_count = cat_df['region'].value_counts()
seaborn.barplot(region_count.index, region_count.values, alpha=0.9)
plot.show()

# Using One-Hot Encoding to tackle categorical data
dummy_df = pd.get_dummies(cat_df)

# Concatenating both original and dummy data
final_df = pd.concat([dataset, dummy_df], axis=1)

# Dropping columns with categorical value
final_df = final_df.drop(['sex', 'smoker', 'region'], axis=1)

# X contains Independent variable and y contains dependent variable
X = final_df[['age', 'sex_male', 'sex_female', 'bmi', 'children', 'smoker_yes', 'smoker_no', 'region_northeast',
              'region_northwest', 'region_southeast', 'region_southwest']].values
y = final_df['expenses'].values

# Splitting data to 80% as training and 20% as test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Initializing LinearRegression object and training it on training data
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Printing Intercept
print('Intercept is : ', regressor.intercept_)

# Getting column names to use while printing co-efficient
column_names = final_df.columns.values[:-1]

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
plot.title('Actual vs Predicted')
plot.grid(which='major', linestyle='-', linewidth='0.2', color='green')
plot.show()

print('Mean absolute error : ', metrics.mean_absolute_error(y_test, y_predict))
print('Mean squared error : ', metrics.mean_squared_error(y_test, y_predict))
print('Root mean squared error : ', np.sqrt(metrics.mean_squared_error(y_test, y_predict)))