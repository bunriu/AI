import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# アヤメデータの読込み
iris_data = pd.read_csv("iris.csv")

# 特徴量データと教師データに分離
X = iris_data.loc[:, ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]].values
y = iris_data.loc[:, "Name"].values

# 学習データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# モデルの学習
estimator = SVC()
estimator.fit(X_train, y_train)

# モデルの評価
y_pred = estimator.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print ("正解率 = ", accuracy)

# 未知データの分類
data = [[4.2, 3.1, 1.6, 0.5]]
X_pred = np.array(data)
y_pred = estimator.predict(X_pred)
print(y_pred)
