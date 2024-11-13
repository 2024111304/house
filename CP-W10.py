import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score, mean_squared_error

# Read the CSV file, skipping the first row and setting the correct column names
data = pd.read_csv(r"C:\Users\Class\Desktop\boston_house_prices.csv", header=1)
print(pd.isnull(data).sum().sum()) # Number of null cells

# Convert all columns to numeric
for i in range(len(data.columns)):
    data.iloc[:, i] = data.iloc[:, i].astype(float)
print(data)
print(data["MEDV"].max()) # Get maximum of house price
print(data["MEDV"].min()) # Get minimum of house price
print(data["MEDV"].mean()) # Get average of house price
print(data["MEDV"].median()) # Get median of house price

plt.hist(data["MEDV"], bins=[0,10,20,30,40,50])
plt.title('Distribution of House Prices')
plt.xlabel('House Price')
plt.ylabel('Frequency')
plt.show()

data['RM'] = data['RM'].round()
grouped_data = data.groupby('RM')
for i in grouped_data.groups:
    plt.bar(i, grouped_data.get_group(i)["MEDV"].mean(), color='blue')
plt.title('Distribution of Boston Housing Prices Group by RM')
plt.xlabel('House Price')
plt.ylabel('Frequency')
plt.show()

x_train, x_test, y_train, y_test = train_test_split(data.drop('MEDV', axis=1), data['MEDV'], test_size=0.2, random_state=0)
model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print('Mean Squared Error:', mse)
print('Root Mean Squared Error:', rmse)
print('R^2:', r2)
