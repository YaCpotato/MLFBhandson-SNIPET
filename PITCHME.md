---

## MLFB機械学習ハンズオン

---

## 本ハンズオンの目的

* とにかくプログラムを動かせるようになること
* 共に頑張る仲間をつくること

---

## 会場諸注意

* 途中退館を希望される場合はスタッフにお申し出ください。
* 写真の撮影を予定しています。撮影NGの方は事前にお申し出ください。
* イベントに関係のない営業や勧誘を目的としたご参加はご遠慮ください。

---

## まずは自己紹介
* 名前
* 意気込み

---

## でははじめましょう

---

## 1. 機械学習ってなに？

---

![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/448316/f75d98e9-3c29-54ae-bda9-28df968b4a8a.png)

---

## 教師あり学習

* 学習するデータの正解がわかる状態で学習




例： 画像分類、物体検出、機械翻訳

---
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/448316/b0ca4d26-4d73-5406-1ade-938438a335f9.png)

---

![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/448316/d3777dec-66d1-6c1d-9a31-cfc35a27fe59.png)


---

![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/448316/53b023d4-503e-4ec4-5f2b-9154815d6fc6.png)


---

![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/448316/31eaf309-3bc6-3b65-e408-400c43f206b8.png)


---

![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/448316/32eae4f3-cbd0-ec43-13d9-392c30c672ff.png)

---

## 教師なし学習

---

## GAN

---

## GANとは
生成モデルの一種. 
データから特徴を学習することで、

* 実在しないデータを生成
* 存在するデータの特徴に沿って変換

できる

---

## Face swapping

---

## 強化学習

---
## 強化学習とは
* ゲームAIなどに採用されているアルゴリズム
* 価値を最大化することを学習する  
<img width="300" alt="go" style="display:inline;" src="https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/448316/8bcd7dbe-e0df-7c1d-448f-7c960d6e4af4.png">
<img width="150" alt="candy" style="display:inline;" src="https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/448316/9e6ab01f-600d-b08b-4f5c-2e18e5c5fe4d.png">

---

## 2. 機械学習ライブラリ

---

## scikit-learn  
https://scikit-learn.org/stable/


---

## 深層学習関連ライブラリ

---

* tensorflow  
https://www.tensorflow.org/
* keras  
https://keras.io/ja/
* pytorch  
https://pytorch.org/
* chainer  
https://chainer.org/

---

## 線形回帰ってなんぞ

---

* 統計学における回帰分析の一種  
回帰分析：  
複数の変数の集合(X,Y)に対して、  
Y=f(X)  
のように関数として表せると仮定し、XからYを見る(予測する)こと

---

## 3. python環境構築
1. Google Colaboratory  
https://colab.research.google.com/notebooks/welcome.ipynb?hl=ja
2. ローカルの実行環境(ある方)  
venv:仮想環境　https://qiita.com/fiftystorm36/items/b2fd47cf32c7694adc2e

---

## 4. Bostonについて分析してみよう


```python

from sklearn.datasets import load_boston
boston = load_boston()
import pandas as pd
boston_df = pd.DataFrame(boston.data, columns = boston.feature_names)
boston_df['MEDV'] = boston.target
boston_df.head()
```
---

![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/448316/4a1ac767-047d-7ef5-3940-f93de9b75105.png)

---

|カラム名|内容|
|:--:|:--:|
| CRIM | 犯罪発生率 |
| ZN | 25,000平方フィート以上の住宅区画の割合 |
| INDUS | 非小売業種の土地面積の割合 |
| CHAS | チャールズ川沿いかを表すダミー変数 |
| NOX | 窒素酸化物の濃度 |
| RM | 平均部屋数 |
| AGE | 1940年より前に建てられた建物の割合 |
| DIS | 5つのボストンの雇用施設への重み付き距離 |
| RAD | 高速道路へのアクセスのしやすさ |
| TAX | 10,000ドルあたりの不動産税率 |
| PTRATIO | 生徒と教師の割合 |
| B | 黒人の割合 |
| LSTAT | 低所得者の割合 |
| MEDV | 住宅価格の中央値（1,000単位） |

---

## とりあえずグラフにしてみる

---

```python
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks")
plt.style.use('seaborn-whitegrid')
plt.scatter(boston_df['RM'], boston_df['MEDV'])
plt.title('Rooms and Prices')
plt.xlabel('Rooms')
plt.ylabel('Prices[K]')
plt.show()
```

---

![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/448316/c5edcb94-5a3d-feef-0f1a-16e56d7f9742.png)

---
## 線形回帰してみる
---

```python
from sklearn.linear_model import LinearRegression
X = boston_df[['RM']].values
Y = boston_df['MEDV'].values
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.7, test_size = 0.3, random_state = 0)
lr = LinearRegression()
lr.fit(X_train, Y_train)
```

---

```python
print('coefficient = ', lr.coef_[0]) 
print('intercept = ', lr.intercept_)
```

---

```python
plt.scatter(X, Y, color = 'blue')         
plt.plot(X, lr.predict(X), color = 'red') 

plt.title('Regression Line')             
plt.xlabel('Average number of rooms [RM]')
plt.ylabel('Prices in $1000\'s [MEDV]')
plt.grid()
plt.show()
```
---

![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/448316/15d28908-1c3c-d4a8-bde5-8a9d5f32bc99.png)


---

```python
from sklearn.metrics import mean_squared_error
Y_pred = lr.predict(X_test)
Y_train_pred = lr.predict(X_train)
print('MSE train data: ', mean_squared_error(Y_train, Y_train_pred))
print('MSE test data: ', mean_squared_error(Y_test, Y_pred))
```

---

```python
from sklearn.metrics import r2_score
print('r^2 train data: ', r2_score(Y_train, Y_train_pred))
print('r^2 test data: ', r2_score(Y_test, Y_pred))
```

---

## 6. (Advanced)回帰分析ってなに？

---

ある変数が与えられた時に、それと相関関係にある値を予測
年齢→年収  
年度→売上

---

## 7. (Advanced)挑戦 その１

画像分類
使用ライブラリ：keras

http://kikei.github.io/ai/2018/03/25/cifer10-cnn1.html

---

## 8. (Advanced)挑戦 その１
自然言語処理

---
スポンサートーク

---
