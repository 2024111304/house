import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Read data
train = pd.read_csv(r"Data/train.csv", sep=';')
test = pd.read_csv(r"Data/test.csv", sep=';')

# Print null values
print(f"Train null values: {train.isnull().sum().sum()}")
print(f"Test null values: {test.isnull().sum().sum()}")

# Convert "y" to numeric values (for convenience later)
train.y.replace(('yes', 'no'), (1, 0), inplace=True)
test.y.replace(('yes', 'no'), (1, 0), inplace=True) 


age_bins = [0,20,40,60,80,100]
train['ageBin'] = pd.cut(train['age'], bins=age_bins) # Create age bins

categories = ['job', 'marital', 'education', 'loan', 'ageBin']
header_text = ["Count of y by Job Type", "Count of y by Marital Status", "Count of y by Education Status", "Count of y by Loan", "Count of y by Age Group"]
for i in range(len(categories)): 
    yes_no_counts = train.groupby([categories[i]] + ['y']).size().unstack().fillna(0) # Group by category and y (into unique combinations) and count
    yes_no_counts.columns = ['no', 'yes'] # Rename columns for plotting yes/no instead of 0/1
    ax = yes_no_counts.plot(kind='barh', stacked=False, figsize=(12, 8), color=['lightblue', 'lightcoral'])
    plt.title(header_text[i])
    plt.xlabel('Count')
    plt.ylabel('Type')
    plt.legend(title='y')

    for container in ax.containers: # Add labels to the bars
        ax.bar_label(container)

    plt.show()

train.loan.replace(('yes', 'no'), (1, 0), inplace=True) # Convert loan to numeric values now that we're done with the plots
test.loan.replace(('yes', 'no'), (1, 0), inplace=True) # This is required for the model to work
selected_columns = ['age', 'balance', 'loan', 'y'] # Select columns for the model (as the exercise requires)
new_train = train[selected_columns]
new_test = test[selected_columns]
x_train, x_test, y_train, y_test = train_test_split(new_train.drop('y', axis=1), new_train['y'], test_size=0.2, random_state=0) # Split the data
model = LogisticRegression() # Create the model
model.fit(x_train, y_train) # Fit the model
y_pred = model.predict(x_test) # Predict
accuracy = accuracy_score(y_test, y_pred) # Calculate accuracy
print(f"Logistic Regression Accuracy: {accuracy:.2f}") # Print accuracy