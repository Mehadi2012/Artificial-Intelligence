import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from google.colab import drive
drive.mount('/content/drive/')

mushroom = pd.read_csv('/content/drive/MyDrive/Mushroom edibility classification/mushroom edibility classification dataset.csv')
mushroom.head(12)

mushroom.shape

mushroom.isnull().sum()

mushroom[['cap-shape','cap-color']]

from sklearn.impute import SimpleImputer

impute = SimpleImputer(missing_values=np.nan, strategy='mean')

impute.fit(mushroom[['cap-shape','cap-color']])

mushroom[['cap-shape','cap-color']] = impute.transform(mushroom[['cap-shape','cap-color']])

mushroom.head(56)

mushroom.info()

mushroom['bruises'].unique()

mushroom['class'].unique()

from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
mushroom['bruises'] = enc.fit_transform(mushroom['bruises'])
mushroom['class'] = enc.fit_transform(mushroom['class'])

print(mushroom[['bruises']].head())
print(mushroom[['class']].head())

mushroom.head(20)

import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(mushroom[['cap-shape','cap-surface','cap-color','bruises','odor','stalk-shape','stalk-root','stalk-surface-above-ring',
                                                                         'stalk-surface-below-ring','stalk-color-above-ring','stalk-color-below-ring','veil-type','veil-color','ring-number','ring-type','spore-print-color',
                                                                         'population','habitat']], mushroom['class'],test_size= 0.2,random_state=2)

print(x_train.shape)
print(x_test.shape)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(x_train)

x_train_scaled = scaler.transform(x_train)

print("per-feature minimum before scaling:\n {}".format(x_train.min(axis=0)))
print("per-feature maximum before scaling:\n {}".format(x_train.max(axis=0)))

print("per-feature minimum after scaling:\n {}".format(
    x_train_scaled.min(axis=0)))
print("per-feature maximum after scaling:\n {}".format(
    x_train_scaled.max(axis=0)))

x_test_scaled = scaler.transform(x_test)

mushroom.head(12)

log_reg_model = LogisticRegression()
log_reg_model.fit(x_train, y_train)
pred = log_reg_model.predict(x_test)
print(pred)

print( accuracy_score(y_test, pred))

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
classif = DecisionTreeClassifier(criterion='entropy',random_state=2)
classif.fit(x_train,y_train)
y_pred = classif.predict(x_test)
dec_tree_score=accuracy_score(y_pred,y_test)
print(dec_tree_score)

x = ['Logistic regression', 'Decision tree']
y = [accuracy_score(y_test, pred),score]
plt.bar(x[0],y[0],color='g')
plt.bar(x[1],y[1],color='y')

plt.show()
