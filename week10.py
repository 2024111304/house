# 匯入必要的套件
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 讀取 Boston 房價資料集
url = 'C:\\Users\\user\\Downloads\\boston_house_prices.csv'
data = pd.read_csv(url, encoding="utf-8", header=None, skiprows=1)

# 設定欄位名稱
data.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'PRICE']

# 檢查是否有缺失值
missing_values = data.isnull().sum()
print("缺失值檢查：\n", missing_values)

# 強制將 PRICE 欄位轉換為數字，無法轉換的值會被設為 NaN
data['PRICE'] = pd.to_numeric(data['PRICE'], errors='coerce')

# 再次檢查是否有缺失值（NaN）
missing_values = data.isnull().sum()
print("\n清理後的缺失值檢查：\n", missing_values)

# 計算並列出最高房價、最低房價、平均房價和中位數房價
max_price = data['PRICE'].max()
min_price = data['PRICE'].min()
mean_price = data['PRICE'].mean()
median_price = data['PRICE'].median()

print(f"\n最高房價: {max_price}")
print(f"最低房價: {min_price}")
print(f"平均房價: {mean_price}")
print(f"中位數房價: {median_price}")

# 繪製房價分布直方圖 (以10為區間)
plt.hist(data['PRICE'], bins=range(int(min_price), int(max_price) + 10, 10), edgecolor='k')
plt.xlabel('House Price')
plt.ylabel('Frequency')
plt.title('Distribution of House Prices')
plt.show()

# 四捨五入 RM 到整數
data['RM_rounded'] = data['RM'].round()

# 使用 groupby 分析Distribution of House Prices
rm_price_mean = data.groupby('RM_rounded')['PRICE'].mean()
print("\nDistribution of House Prices：\n", rm_price_mean)

# 繪製 RM 值與平均房價的直方圖
rm_price_mean.plot(kind='bar', edgecolor='k')
plt.xlabel('House Price')
plt.ylabel('Frequency')
plt.title('Distribution of House Prices')
plt.show()

# 建立線性回歸模型並進行房價預測
X = data[['RM']]  # 特徵選擇
y = data['PRICE']  # 標籤選擇

# 分割訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 建立並訓練線性回歸模型
model = LinearRegression()
model.fit(X_train, y_train)

# 使用測試集進行預測
y_pred = model.predict(X_test)

# 計算均方誤差 (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"\n線性回歸模型的均方誤差: {mse}")
