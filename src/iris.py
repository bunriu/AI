import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# �A�����f�[�^�̓Ǎ���
iris_data = pd.read_csv("iris.csv")

# �����ʃf�[�^�Ƌ��t�f�[�^�ɕ���
X = iris_data.loc[:, ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]].values
y = iris_data.loc[:, "Name"].values

# �w�K�f�[�^�ƃe�X�g�f�[�^�ɕ���
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# ���f���̊w�K
estimator = SVC()
estimator.fit(X_train, y_train)

# ���f���̕]��
y_pred = estimator.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print ("���� = ", accuracy)

# ���m�f�[�^�̕���
data = [[4.2, 3.1, 1.6, 0.5]]
X_pred = np.array(data)
y_pred = estimator.predict(X_pred)
print(y_pred)
