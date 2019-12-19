import pandas as pd
import numpy as np
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
df = pd.read_csv('train.csv', names=['age', 'job', 'marital', 'education', 'housing', 'loan', 'y'])
ori_df = df.copy()

print('**' * 50)
print(df.nunique())
print('**' * 50)

cols = ['age', 'job', 'marital', 'education', 'housing', 'loan']
category_cols = ['job', 'marital', 'housing', 'education', 'loan']
notused_cols = []
continuous_cols = ['age']

target = 'y'

x = pd.concat([df[category_cols], df[continuous_cols]], axis=1)
y = df[target]

# 离散特征one-hot编码
for col in category_cols:
    onehot_feats = pd.get_dummies(x[col], prefix=col)
    x.drop([col], axis=1, inplace=True)
    x = pd.concat([x, onehot_feats], axis=1)


def model_build():
    # 数据集分割
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    # 建立RandomForestClassifier模型
    # model = RandomForestClassifier(n_estimators=100)
    # model.fit(x_train, y_train)
    # y_pred = model.predict(x_test)
    # result = classification_report(y_test, y_pred)
    # print(result)

    # 建立XGBClassifier模型
    # model = XGBClassifier(n_estimators=200)
    # eval_set = [(x_test, y_test)]
    # model.fit(x_train, y_train, early_stopping_rounds=50, eval_metric="auc", eval_set=eval_set, verbose=True)
    # y_pred = model.predict(x_test)
    # result = classification_report(y_test, y_pred)
    # print(result)

    # 建立BaggingClassifier模型
    model = BaggingClassifier(n_estimators=40, random_state=0)
    model.fit(x_train, y_train)
    y_pred = model.predict(x)
    result = classification_report(y, y_pred)
    print(result)


if __name__ == '__main__':
    model_build()
