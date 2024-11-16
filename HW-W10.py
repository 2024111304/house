import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# 讀取數據
train = pd.read_csv(r"Data/train.csv", sep=';') # 讀取訓練數據
test = pd.read_csv(r"Data/test.csv", sep=';') # 讀取測試數據

# 檢查是否有缺失值
print(f"Train null values: {train.isnull().sum().sum()}") # 計算訓練數據中的缺失值數量
print(f"Test null values: {test.isnull().sum().sum()}") # 計算測試數據中的缺失值數量

# 將目標變數 "y" 轉換為數字值（便於後續分析）
train.y.replace(('yes', 'no'), (1, 0), inplace=True) # 訓練數據中的 "yes" 替換為 1，"no" 替換為 0
test.y.replace(('yes', 'no'), (1, 0), inplace=True)  # 測試數據同樣處理

# 年齡分組
age_bins = [0,20,40,60,80,100] # 定義年齡分組區間
train['ageBin'] = pd.cut(train['age'], bins=age_bins) # 為訓練數據創建年齡分組

# 定義類別變數及標題
categories = ['job', 'marital', 'education', 'loan', 'ageBin'] # 定義需要分析的類別變數
header_text = ["Count of y by Job Type", "Count of y by Marital Status", "Count of y by Education Status", "Count of y by Loan", "Count of y by Age Group"] # 標題文字

# 繪製每個類別變數的直方圖
for i in range(len(categories)): 
    yes_no_counts = train.groupby([categories[i]] + ['y']).size().unstack().fillna(0) # 按類別和 y 分組並計數
    yes_no_counts.columns = ['no', 'yes'] # 重命名列，顯示為 "yes"/"no" 而非數字
    ax = yes_no_counts.plot(kind='barh', stacked=False, figsize=(12, 8), color=['lightblue', 'lightcoral']) # 繪製水平直方圖
    plt.title(header_text[i])  # 設置標題
    plt.xlabel('Count') # 設置 x 軸標籤
    plt.ylabel('Type') # 設置 y 軸標籤
    plt.legend(title='y') # 添加圖例

    for container in ax.containers: # 在每個柱狀上顯示數值
        ax.bar_label(container)

    plt.show()

# 將貸款欄位轉換為數字值
train.loan.replace(('yes', 'no'), (1, 0), inplace=True) # 訓練數據的 "yes" 替換為 1，"no" 替換為 0
test.loan.replace(('yes', 'no'), (1, 0), inplace=True) # 測試數據同樣處理
# 選擇建模需要的欄位
selected_columns = ['age', 'balance', 'loan', 'y'] # 定義需要的列
new_train = train[selected_columns] # 過濾訓練數據
new_test = test[selected_columns] # 過濾測試數據
# 分割數據為訓練集和測試集
x_train, x_test, y_train, y_test = train_test_split(new_train.drop('y', axis=1), new_train['y'], test_size=0.2, random_state=0) # 分割數據
# 建立並訓練邏輯回歸模型
model = LogisticRegression() # 創建邏輯回歸模型
model.fit(x_train, y_train) # 訓練模型
# 預測及計算準確率
y_pred = model.predict(x_test) # 用測試集進行預測
accuracy = accuracy_score(y_test, y_pred) # 計算準確率
print(f"Logistic Regression Accuracy: {accuracy:.2f}") # 印出準確率