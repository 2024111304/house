import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


train = pd.read_csv(r"C:\Users\Class\Desktop\train.csv", sep=';')
test = pd.read_csv(r"C:\Users\Class\Desktop\test.csv", sep=';')

print(f"Train null values: {train.isnull().sum().sum()}")
print(f"Test null values: {test.isnull().sum().sum()}")
train.y.replace(('yes', 'no'), (1, 0), inplace=True)
test.y.replace(('yes', 'no'), (1, 0), inplace=True) 

age_bins = [0,20,40,60,80,100]
train['ageBin'] = pd.cut(train['age'], bins=age_bins)

categories = ['job', 'marital', 'education', 'loan', 'ageBin']
for category in categories:
    yes_no_counts = train.groupby([category] + ['y']).size().unstack().fillna(0)
    yes_no_counts.columns = ['no', 'yes']
    ax = yes_no_counts.plot(kind='barh', stacked=False, figsize=(12, 8), color=['lightblue', 'lightcoral'])
    plt.title('Number of Yes and No for y based on different categories')
    plt.xlabel('Count')
    plt.ylabel('Categories')
    plt.legend(title='y')

    for container in ax.containers:
        ax.bar_label(container)

    plt.show()

train.loan.replace(('yes', 'no'), (1, 0), inplace=True)
test.loan.replace(('yes', 'no'), (1, 0), inplace=True)
selected_columns = ['age', 'balance', 'loan', 'y']
new_train = train[selected_columns]
new_test = test[selected_columns]
x_train, x_test, y_train, y_test = train_test_split(new_train.drop('y', axis=1), new_train['y'], test_size=0.2, random_state=0)
model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test) 
accuracy = accuracy_score(y_test, y_pred)
print(f"Logistic Regression Accuracy: {accuracy:.2f}")