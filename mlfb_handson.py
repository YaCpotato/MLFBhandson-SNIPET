# -*- coding: utf-8 -*-

from sklearn.datasets import load_boston
boston = load_boston() 															# データセットの読み込み
import pandas as pd
boston_df = pd.DataFrame(boston.data, columns = boston.feature_names) 			# 説明変数(boston.data)をDataFrameに保存
boston_df['MEDV'] = boston.target 												# 目的変数(boston.target)もDataFrameに追加
boston_df.head()

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks")
plt.style.use('seaborn-whitegrid')
plt.scatter(boston_df['RM'], boston_df['MEDV'], color = 'blue')					# 平均部屋数と住宅価格の散布図をプロット
plt.title('Rooms and Prices')													# 図のタイトル
plt.xlabel('Rooms') 															# x軸のラベル
plt.ylabel('Prices[K]')    														# y軸のラベル
plt.savefig('number_of_rooms__prices.png')																		# 図の表示

from sklearn.linear_model import LinearRegression
X = boston_df[['RM']].values         											# 説明変数（Numpyの配列）
Y = boston_df['MEDV'].values         											# 目的変数（Numpyの配列）
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.7, test_size = 0.3, random_state = 0) 
																				# データを学習用と検証用に分割
lr = LinearRegression()
lr.fit(X_train, Y_train) 														# 線形モデルの重みを学習

print('coefficient = ', lr.coef_[0]) 											# 説明変数の係数を出力
print('intercept = ', lr.intercept_) 											# 切片を出力

plt.style.use('seaborn-whitegrid')
plt.scatter(X, Y, color = 'blue')         										# 説明変数と目的変数のデータの散布図をプロット
plt.plot(X, lr.predict(X), color = 'red') 										# 回帰直線をプロット
plt.title('Regression Line')               										# 図のタイトル
plt.xlabel('Rooms') 															# x軸のラベル
plt.ylabel('Prices[K]')    														# y軸のラベル
plt.grid()                                 										# グリッド線を表示
plt.savefig('number_of_rooms__prices_withLR.png')        						# 図の表示

from sklearn.metrics import mean_squared_error
Y_pred = lr.predict(X_test) 													# 検証データを用いて目的変数を予測
Y_train_pred = lr.predict(X_train) 												# 学習データに対する目的変数を予測
print('学習時の平均二乗誤差: ', mean_squared_error(Y_train, Y_train_pred)) 		# 学習データを用いたときの平均二乗誤差を出力
print('検証時の平均二乗誤差: ', mean_squared_error(Y_test, Y_pred))         	# 検証データを用いたときの平均二乗誤差を出力
from sklearn.metrics import r2_score
print('学習時の決定係数: ', r2_score(Y_train, Y_train_pred))
print('検証時の決定係数: ', r2_score(Y_test, Y_pred))

