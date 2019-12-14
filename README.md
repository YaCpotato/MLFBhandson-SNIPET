# MLFBハンズオンREADME兼スニペット

## pythonのインストール
python公式サイト
https://www.python.org/downloads/release/python-373/ から
python3.7.3をインストール↓

**インストール時、 add python to my PATH にチェックをする**

### Mac
[macOS 64-bit installer](https://www.python.org/ftp/python/3.7.3/python-3.7.3-macosx10.9.pkg)
### Windows
[Windows x86-64 executable installer](https://www.python.org/ftp/python/3.7.3/python-3.7.3-amd64.exe)

## pythonの環境構築
**Windowsはコマンドプロンプトを管理者権限で開く**
Userディレクトリ直下へ移動し、pythonの仮想環境を作る
### Mac & Windows

```
python3 -m venv mlfb
```


### Windows

```
python -m venv mlfb
```

## ライブラリのインストール

```
pip install -r "requirements.txt"
```

requirements.txtは下記

```
cycler==0.10.0
joblib==0.14.1
kiwisolver==1.1.0
matplotlib==3.1.2
numpy==1.17.4
pandas==0.25.3
pyparsing==2.4.5
python-dateutil==2.8.1
pytz==2019.3
scikit-learn==0.22
scipy==1.3.3
seaborn==0.9.0
six==1.13.0
sklearn==0.0
```

## プログラムの実行

```
python mlfb_handson.py
```

# ソースコードのスニペット

```python
from sklearn.datasets import load_boston
boston = load_boston() 															# データセットの読み込み
import pandas as pd
boston_df = pd.DataFrame(boston.data, columns = boston.feature_names) 			# 説明変数(boston.data)をDataFrameに保存
boston_df['MEDV'] = boston.target 												# 目的変数(boston.target)もDataFrameに追加
boston_df.head()
```

```python
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks")
plt.style.use('seaborn-whitegrid')
plt.scatter(boston_df['RM'], boston_df['MEDV'], color = 'blue')					# 平均部屋数と住宅価格の散布図をプロット
plt.title('Rooms and Prices')													# 図のタイトル
plt.xlabel('Rooms') 															# x軸のラベル
plt.ylabel('Prices[K]')    														# y軸のラベル
plt.savefig('number_of_rooms__prices.png')	
```

```python
from sklearn.linear_model import LinearRegression
X = boston_df[['RM']].values         											# 説明変数（Numpyの配列）
Y = boston_df['MEDV'].values         											# 目的変数（Numpyの配列）
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.7, test_size = 0.3, random_state = 0) 
																				# データを学習用と検証用に分割
lr = LinearRegression()
lr.fit(X_train, Y_train)
```

```python
print('coefficient = ', lr.coef_[0]) 											# 説明変数の係数を出力
print('intercept = ', lr.intercept_) 											# 切片を出力
```

```python
plt.style.use('seaborn-whitegrid')
plt.scatter(X, Y, color = 'blue')         										# 説明変数と目的変数のデータの散布図をプロット
plt.plot(X, lr.predict(X), color = 'red') 										# 回帰直線をプロット
plt.title('Regression Line')               										# 図のタイトル
plt.xlabel('Rooms') 															# x軸のラベル
plt.ylabel('Prices[K]')    														# y軸のラベル
plt.grid()                                 										# グリッド線を表示
plt.savefig('number_of_rooms__prices_withLR.png')        						# 図の表示
```

```python
from sklearn.metrics import mean_squared_error
Y_pred = lr.predict(X_test) 													# 検証データを用いて目的変数を予測
Y_train_pred = lr.predict(X_train) 												# 学習データに対する目的変数を予測
print('学習時の平均二乗誤差: ', mean_squared_error(Y_train, Y_train_pred)) 		# 学習データを用いたときの平均二乗誤差を出力
print('検証時の平均二乗誤差: ', mean_squared_error(Y_test, Y_pred))         	# 検証データを用いたときの平均二乗誤差を出力
from sklearn.metrics import r2_score
print('学習時の決定係数: ', r2_score(Y_train, Y_train_pred))
print('検証時の決定係数: ', r2_score(Y_test, Y_pred))
```
